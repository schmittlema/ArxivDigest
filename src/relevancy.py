"""
run:
python -m relevancy run_all_day_paper \
  --model_name="gpt-3.5-turbo-16k" \
"""
import time
import json
import os
import random
import re
import string
from datetime import datetime

import numpy as np
import tqdm
import utils

from paths import DATA_DIR


def encode_prompt(query, prompt_papers):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("src/relevancy_prompt.txt").read() + "\n"
    prompt += query['interest']

    for idx, task_dict in enumerate(prompt_papers):
        (title, authors, abstract, content) = task_dict["title"], task_dict["authors"], task_dict["abstract"], task_dict["content"]
        if not title:
            raise
        prompt += f"###\n"
        prompt += f"{idx + 1}. Title: {title}\n"
        prompt += f"{idx + 1}. Authors: {authors}\n"
        prompt += f"{idx + 1}. Abstract: {abstract}\n"
        prompt += f"{idx + 1}. Content: {content}\n"
    prompt += f"\n Generate response:\n1."
    return prompt


def is_json(myjson):
    try:
        json.loads(myjson)
    except Exception as e:
        return False
    return True

def post_process_chat_gpt_response(paper_data, response, threshold_score=7):
    selected_data = []
    if response is None:
        return []
        
    # Handle both old and new API response formats
    if isinstance(response, dict) and 'message' in response:
        # Old API format
        content = response['message']['content']
    elif hasattr(response, 'choices') and len(response.choices) > 0:
        # New API format (OpenAI Client)
        content = response.choices[0].message.content
    else:
        # Fallback to dictionary access
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        except Exception:
            content = ''
            
    if not content:
        return [], False
        
    json_items = content.replace("\n\n", "\n").split("\n")
    pattern = r"^\d+\. |\\"
    import pprint

    def try_loads(line):
        try:
            return json.loads(re.sub(pattern, "", line))
        except json.JSONDecodeError:
            return None
            
    score_items = []
    try:
        for line in json_items:
            if is_json(line) and "relevancy score" in line.lower():
                score_items.append(json.loads(re.sub(pattern, "", line)))
    except Exception as e:
        pprint.pprint([re.sub(pattern, "", line) for line in json_items if "relevancy score" in line.lower()])
        try:
            score_items = score_items[:-1]
        except Exception:
            score_items = []
        print(e)
        raise RuntimeError("failed")
        
    pprint.pprint(score_items)
    scores = []
    for item in score_items:
        temp = item["Relevancy score"]
        if isinstance(temp, str) and "/" in temp:
            scores.append(int(temp.split("/")[0]))
        else:
            scores.append(int(temp))
            
    if len(score_items) != len(paper_data):
        score_items = score_items[:len(paper_data)]
        hallucination = True
    else:
        hallucination = False

    for idx, inst in enumerate(score_items):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if scores[idx] < threshold_score:
            continue
        output_str = "Subject: " + paper_data[idx]["subjects"] + "\n"
        output_str += "Title: " + paper_data[idx]["title"] + "\n"
        output_str += "Authors: " + paper_data[idx]["authors"] + "\n"
        output_str += "Link: " + paper_data[idx]["main_page"] + "\n"
        for key, value in inst.items():
            paper_data[idx][key] = value
            output_str += str(key) + ": " + str(value) + "\n"
        paper_data[idx]['summarized_text'] = output_str
        selected_data.append(paper_data[idx])
        
    return selected_data, hallucination


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def process_subject_fields(subjects):
    all_subjects = subjects.split(";")
    all_subjects = [s.split(" (")[0] for s in all_subjects]
    return all_subjects

def generate_relevance_score(
    all_papers,
    query,
    model_name="gpt-3.5-turbo-16k",
    threshold_score=7,
    num_paper_in_prompt=1,
    temperature=0.4,
    top_p=1.0,
    sorting=True
):
    ans_data = []
    request_idx = 1
    hallucination = False
    for id in tqdm.tqdm(range(0, len(all_papers), num_paper_in_prompt)):
        prompt_papers = all_papers[id:id+num_paper_in_prompt]
        # only sampling from the seed tasks
        prompt = encode_prompt(query, prompt_papers)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=1024*num_paper_in_prompt, # The response for each paper should be less than 128 tokens.
            top_p=top_p,
        )
        request_start = time.time()
        response = utils.openai_completion(
            prompts=prompt,
            model_name=model_name,
            batch_size=1,
            decoding_args=decoding_args,
            logit_bias={"100257": -100},  # prevent the <|endoftext|> from being generated
        )
        print ("response", response['message']['content'])
        request_duration = time.time() - request_start

        process_start = time.time()
        batch_data, hallu = post_process_chat_gpt_response(prompt_papers, response, threshold_score=threshold_score)
        hallucination = hallucination or hallu
        ans_data.extend(batch_data)

        print(f"Request {request_idx+1} took {request_duration:.2f}s")
        print(f"Post-processing took {time.time() - process_start:.2f}s")

    if sorting:
        ans_data = sorted(ans_data, key=lambda x: int(x["Relevancy score"]), reverse=True)
    
    return ans_data, hallucination

def run_all_day_paper(
    query={"interest":"Computer Science", "subjects":["Machine Learning", "Computation and Language", "Artificial Intelligence", "Information Retrieval"]},
    date=None,
    model_name="gpt-3.5-turbo-16k",
    threshold_score=7,
    num_paper_in_prompt=2,
    temperature=0.4,
    top_p=1.0
):
    if date is None:
        date = datetime.today().strftime('%a, %d %b %y')
        # string format such as Wed, 10 May 23
    print ("the date for the arxiv data is: ", date)

    file_path = os.path.join(DATA_DIR, f"{date}.jsonl")
    all_papers = [json.loads(l) for l in open(file_path, "r")]
    print (f"We found {len(all_papers)}.")

    all_papers_in_subjects = [
        t for t in all_papers
        if bool(set(process_subject_fields(t['subjects'])) & set(query['subjects']))
    ]
    print(f"After filtering subjects, we have {len(all_papers_in_subjects)} papers left.")
    ans_data = generate_relevance_score(all_papers_in_subjects, query, model_name, threshold_score, num_paper_in_prompt, temperature, top_p)
    from paths import DIGEST_DIR
    utils.write_ans_to_file(ans_data, date, output_dir=DIGEST_DIR)
    return ans_data


if __name__ == "__main__":
    query = {"interest":"""
    1. Large language model pretraining and finetunings
    2. Multimodal machine learning
    3. Do not care about specific application, for example, information extraction, summarization, etc.
    4. Not interested in paper focus on specific languages, e.g., Arabic, Chinese, etc.\n""",
    "subjects":["Computation and Language"]}
    ans_data = run_all_day_paper(query)
