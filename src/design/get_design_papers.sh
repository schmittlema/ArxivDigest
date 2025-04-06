#!/bin/bash
# Simple script to run the design papers crawler

# Default values
DAYS=7
OUTPUT="design_papers.json"
EMAIL=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --days)
      DAYS="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --email)
      EMAIL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the crawler
echo "Searching for design papers from the last $DAYS days..."
python src/design/find_design_papers.py --days "$DAYS" --output "./data/$OUTPUT" --html "./digest/${OUTPUT%.json}.html"

# If email is provided, send the results
if [ -n "$EMAIL" ]; then
  echo "Sending results to $EMAIL..."
  
  # Check if the file exists
  if [ -f "./data/$OUTPUT" ]; then
    # Convert JSON to HTML
    HTML_OUTPUT="./digest/${OUTPUT%.json}.html"
    python -c "
import json
import sys

# Read JSON file
with open('./data/$OUTPUT', 'r') as f:
    papers = json.load(f)

# Create HTML
html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Design Automation Papers</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .paper { margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }
        .title { font-size: 18px; font-weight: bold; color: #2c3e50; }
        .authors { font-style: italic; margin: 5px 0; }
        .categories { color: #3498db; margin-bottom: 10px; }
        .abstract { margin-top: 10px; }
        .techniques { margin-top: 10px; color: #16a085; }
        .metrics { margin-top: 5px; color: #8e44ad; }
        a { color: #2980b9; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Design Automation Papers</h1>
    <p>Found ${len(papers)} papers related to graphic design automation with AI/ML</p>
'''

# Add papers
for paper in papers:
    html += f'''
    <div class='paper'>
        <div class='title'><a href='{paper.get("main_page", "#")}'>{paper.get("title", "No title")}</a></div>
        <div class='authors'>{paper.get("authors", "Unknown authors")}</div>
        <div class='categories'>Category: {paper.get("design_category", "General")} | Subject: {paper.get("subjects", "N/A")}</div>
        <div class='techniques'>Techniques: {', '.join(paper.get("design_techniques", ["None identified"]))}</div>
        <div class='metrics'>Evaluation metrics: {', '.join(paper.get("design_metrics", ["None identified"]))}</div>
        <div class='abstract'><strong>Abstract:</strong> {paper.get("abstract", "No abstract available")}</div>
    </div>
    '''

html += '''
</body>
</html>
'''

# Ensure directory exists
import os
os.makedirs(os.path.dirname('$HTML_OUTPUT'), exist_ok=True)

# Write HTML file
with open('$HTML_OUTPUT', 'w') as f:
    f.write(html)

print(f'Created HTML report at {sys.argv[1]}')
    " "$HTML_OUTPUT"
    
    # Send email (requires sendmail)
    if command -v sendmail &> /dev/null; then
      (
        echo "To: $EMAIL"
        echo "From: design-papers@arxivdigest.local"
        echo "Subject: Design Automation Papers Report"
        echo "Content-Type: text/html"
        echo ""
        cat "$HTML_OUTPUT"
      ) | sendmail -t
      echo "Email sent to $EMAIL"
    else
      echo "Error: sendmail not found. Cannot send email."
      echo "You can view the HTML report at $HTML_OUTPUT"
    fi
  else
    echo "Error: Output file $OUTPUT not found."
  fi
fi

echo "Done!"