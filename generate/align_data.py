"""
Data Alignment Script
=====================
Aligns input and target files using sliding window similarity matching.

Handles cases where lines may be skipped or misaligned by searching ahead
in the target file for the best match.
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher


def extract_words_and_numbers(text):
    """Extract words and numbers from text for comparison"""
    # Remove XML tags for target comparison
    text_clean = re.sub(r'<[^>]+>', '', text)
    # Extract alphanumeric tokens
    tokens = re.findall(r'\w+', text_clean.lower())
    return set(tokens)


def calculate_similarity(input_text, target_text):
    """Calculate similarity score between input and target"""
    input_tokens = extract_words_and_numbers(input_text)
    target_tokens = extract_words_and_numbers(target_text)
    
    if not input_tokens or not target_tokens:
        return 0.0
    
    # Jaccard similarity (intersection over union)
    intersection = len(input_tokens & target_tokens)
    union = len(input_tokens | target_tokens)
    
    return intersection / union if union > 0 else 0.0


def align_with_sliding_window(input_lines, target_lines, window_size=5, threshold=0.3):
    """
    Align input lines with target lines using sliding window search.
    
    Args:
        input_lines: List of input address strings
        target_lines: List of target tag strings
        window_size: How many lines ahead to search for matches
        threshold: Minimum similarity score to consider a match
    
    Returns:
        List of aligned {"input": "...", "target": "..."} dicts
    """
    aligned_data = []
    target_idx = 0
    
    print(f"Aligning {len(input_lines)} inputs with {len(target_lines)} targets...")
    print(f"Window size: {window_size}, Similarity threshold: {threshold}")
    
    for i, input_line in enumerate(input_lines):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(input_lines)} lines...")
        
        # Clean input line: strip whitespace and remove trailing dot
        input_line = input_line.strip()
        if input_line.endswith('.'):
            input_line = input_line[:-1].strip()
        
        if not input_line:
            continue
        
        # Try to find best match within window
        best_match_idx = target_idx
        best_similarity = 0.0
        
        # Search within sliding window
        search_end = min(target_idx + window_size, len(target_lines))
        for j in range(target_idx, search_end):
            target_line = target_lines[j].strip()
            if not target_line:
                continue
            
            similarity = calculate_similarity(input_line, target_line)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = j
        
        # Use the best match if it's above threshold
        if best_similarity >= threshold and best_match_idx < len(target_lines):
            target_line = target_lines[best_match_idx].strip()
            aligned_data.append({
                "input": input_line,
                "target": target_line,
                "llm_mess": True
            })
            # Move target index to next line after the match
            target_idx = best_match_idx + 1
        else:
            # No good match found, try direct alignment
            if target_idx < len(target_lines):
                target_line = target_lines[target_idx].strip()
                aligned_data.append({
                    "input": input_line,
                    "target": target_line,
                    "llm_mess": True
                })
                target_idx += 1
                print(f"  Warning: Line {i+1} matched with low similarity ({best_similarity:.2f})")
            else:
                print(f"  Warning: Line {i+1} has no corresponding target (out of targets)")
    
    return aligned_data


def main():
    ROOT = Path(__file__).resolve().parents[1]
    
    input_file = ROOT / "generate" / "gen_for_llm_inject.jsonl"
    target_file = ROOT / "generate" / "gen_for_llm_targets.jsonl"
    output_file = ROOT / "generate" / "aligned_dataset.jsonl"
    
    # Read input lines
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_lines = f.readlines()
    
    # Read target lines
    print(f"Reading {target_file}...")
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()
    
    print(f"\nInput lines: {len(input_lines)}")
    print(f"Target lines: {len(target_lines)}")
    print(f"Difference: {abs(len(input_lines) - len(target_lines))} lines\n")
    
    # Align data
    aligned_data = align_with_sliding_window(
        input_lines, 
        target_lines, 
        window_size=5,      # Look ahead up to 5 lines
        threshold=0.3       # Minimum 30% similarity
    )
    
    # Save aligned dataset
    print(f"\nSaving aligned dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in aligned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Alignment complete!")
    print(f"  Aligned pairs: {len(aligned_data)}")
    print(f"  Output saved to: {output_file}")
    
    # Show statistics
    if len(aligned_data) < len(input_lines):
        print(f"\n  Warning: {len(input_lines) - len(aligned_data)} input lines were not aligned")


if __name__ == "__main__":
    main()
