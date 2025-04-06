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
    
    # Just log the number of papers for information
    num_papers = len(prompt_papers)
    print(f"Sending prompt with {num_papers} papers for analysis")
    
    return prompt


def is_json(myjson):
    try:
        json.loads(myjson)
    except Exception as e:
        return False
    return True

def extract_json_from_string(text):
    """
    Attempt to extract JSON from a string by finding '{'...'}'
    """
    # Find the outermost JSON object
    stack = []
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{' and start_idx == -1:
            start_idx = i
            stack.append(char)
        elif char == '{':
            stack.append(char)
        elif char == '}' and stack:
            stack.pop()
            if not stack and start_idx != -1:
                # Found complete JSON object
                json_str = text[start_idx:i+1]
                try:
                    parsed = json.loads(json_str)
                    return parsed
                except json.JSONDecodeError:
                    # If this one fails, continue looking
                    start_idx = -1
    
    return None

def post_process_chat_gpt_response(paper_data, response, threshold_score=0):
    """
    Completely rewritten parsing function that handles the OpenAI response better
    """
    selected_data = []
    if response is None:
        print("Response is None")
        return [], False
        
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
        print("Content is empty")
        return [], False
    
    # Print the raw content for debugging
    print(f"\nRaw content:\n{content}\n")
    
    # Try to extract JSON directly from the content
    analysis = extract_json_from_string(content)
    if analysis and "Relevancy score" in analysis:
        score_items = [analysis]
    else:
        # Fallback to older parsing method
        score_items = []
        json_items = content.replace("\n\n", "\n").split("\n")
        pattern = r"^\d+\. |\\"
        
        for line in json_items:
            if is_json(line) and "relevancy score" in line.lower():
                try:
                    parsed_item = json.loads(re.sub(pattern, "", line))
                    score_items.append(parsed_item)
                except:
                    pass
    
    print(f"Found {len(score_items)} score items from response")
    
    # If we have no score items but have paper data, create default ones
    if len(score_items) == 0 and len(paper_data) > 0:
        print("Creating default score items for each paper")
        score_items = []
        for i in range(len(paper_data)):
            # Create a default item with a mid-range score
            score_items.append({
                "Relevancy score": 5,
                "Reasons for match": "Default score assigned due to parsing issues.",
                "Key innovations": "Not available in analysis",
                "Critical analysis": "Not available in analysis",
                "Goal": "Not available in analysis",
                "Data": "Not available in analysis",
                "Methodology": "Not available in analysis",
                "Implementation details": "Not available in analysis",
                "Experiments & Results": "Not available in analysis",
                "Git": "Not available in analysis",
                "Discussion & Next steps": "Not available in analysis",
                "Related work": "Not available in analysis",
                "Practical applications": "Not available in analysis",
                "Key takeaways": "Not available in analysis"
            })
    
    # Truncate score_items if needed
    if len(score_items) > len(paper_data):
        print(f"WARNING: More score items ({len(score_items)}) than papers ({len(paper_data)})")
        score_items = score_items[:len(paper_data)]
        hallucination = True
    else:
        hallucination = False

    # Define expected analysis fields we want to ensure are copied to the paper objects
    analysis_fields = [
        "Relevancy score", "Reasons for match", "Key innovations", "Critical analysis",
        "Goal", "Data", "Methodology", "Implementation details", "Experiments & Results",
        "Git", "Discussion & Next steps", "Related work", "Practical applications", 
        "Key takeaways"
    ]

    print(f"DEBUG: Processing {len(score_items)} score items for {len(paper_data)} papers")
    
    # If we don't have any score items but have papers, something went wrong with parsing
    if len(score_items) == 0 and len(paper_data) > 0:
        print("WARNING: No score items were found, but papers exist. Check JSON parsing.")
        # Create fallback score items with default score to prevent empty results
        for i in range(len(paper_data)):
            fallback_item = {
                "Relevancy score": threshold_score,  # Set to threshold score to ensure it passes filter
                "Reasons for match": "Automatically assigned threshold score due to parsing issues."
            }
            score_items.append(fallback_item)
            
    # Ensure we have at least one paper if there are score items
    for idx, inst in enumerate(score_items):
        if idx >= len(paper_data):
            print(f"DEBUG: Index {idx} out of range for paper_data (length {len(paper_data)})")
            continue
        
        # Get the relevancy score
        relevancy_score = inst.get('Relevancy score', 0)
        if isinstance(relevancy_score, str):
            try:
                # Try to convert string score to integer
                if '/' in relevancy_score:
                    relevancy_score = int(relevancy_score.split('/')[0])
                else:
                    relevancy_score = int(relevancy_score)
            except (ValueError, TypeError):
                relevancy_score = threshold_score  # Default to threshold if conversion fails
        
        print(f"DEBUG: Processing paper {idx+1} with score {relevancy_score}")
        
        # Only process papers that meet the threshold
        if relevancy_score < threshold_score:
            print(f"DEBUG: Skipping paper {idx+1} with score {relevancy_score} < threshold {threshold_score}")
            continue
            
        # Create detailed output string for logging and console display
        output_str = "Subject: " + paper_data[idx]["subjects"] + "\n"
        output_str += "Title: " + paper_data[idx]["title"] + "\n"
        output_str += "Authors: " + paper_data[idx]["authors"] + "\n"
        output_str += "Link: " + paper_data[idx]["main_page"] + "\n"
        
        # Copy all fields from the analysis to the paper object
        for key, value in inst.items():
            paper_data[idx][key] = value
            output_str += str(key) + ": " + str(value) + "\n"
            
        # Ensure all expected analysis fields are present in the paper object
        # This ensures fields used in the HTML template like "Key innovations" are set
        for field in analysis_fields:
            if field in inst:
                # Double-check the field got copied (should be redundant with the loop above)
                paper_data[idx][field] = inst[field]
                print(f"Found and copied field: {field}")
            else:
                print(f"Missing analysis field: {field}")
                paper_data[idx][field] = "Not available in analysis"
        
        paper_data[idx]['summarized_text'] = output_str
        selected_data.append(paper_data[idx])
        print(f"DEBUG: Added paper {idx+1} to selected_data (now has {len(selected_data)} papers)")
        
    print(f"DEBUG: Selected papers count: {len(selected_data)}")
    print(f"DEBUG: Paper fields: {list(selected_data[0].keys()) if selected_data else 'No papers'}")
    
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
    threshold_score=2,
    num_paper_in_prompt=10,  # Default to 10 papers per prompt for better comparative analysis
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
