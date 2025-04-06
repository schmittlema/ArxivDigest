#!/bin/bash
# Wrapper script to run the enhanced design papers finder

# Default values
DAYS=7
OUTPUT=""  # Let the Python script add date to the filename
HTML=""    # Let the Python script add date to the filename
KEYWORD=""
ANALYZE=false
INTEREST=""
MODEL="gpt-3.5-turbo-16k"
NO_DATE=false

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
    --html)
      HTML="$2"
      shift 2
      ;;
    --keyword)
      KEYWORD="$2"
      shift 2
      ;;
    --analyze)
      ANALYZE=true
      shift
      ;;
    --interest)
      INTEREST="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --no-date)
      NO_DATE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the command
CMD="python -m src.design.find_design_papers --days $DAYS"

# Add output and HTML paths if specified
if [ -n "$OUTPUT" ]; then
  CMD="$CMD --output \"$OUTPUT\""
fi

if [ -n "$HTML" ]; then
  CMD="$CMD --html \"$HTML\""
fi

# Add optional arguments if provided
if [ -n "$KEYWORD" ]; then
  CMD="$CMD --keyword \"$KEYWORD\""
fi

if [ "$ANALYZE" = true ]; then
  CMD="$CMD --analyze"
  
  if [ -n "$INTEREST" ]; then
    CMD="$CMD --interest \"$INTEREST\""
  fi
  
  if [ -n "$MODEL" ]; then
    CMD="$CMD --model \"$MODEL\""
  fi
fi

if [ "$NO_DATE" = true ]; then
  CMD="$CMD --no-date"
fi

# Print the command
echo "Running: $CMD"

# Execute the command
eval $CMD

echo "Done!"