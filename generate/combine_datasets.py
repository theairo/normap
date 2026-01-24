"""
Combine Multiple JSONL Datasets
================================
Combines alg_dat.jsonl and final_training_dataset.jsonl into a single dataset.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATE_PATH = ROOT / "generate"

# Input files
INPUT_FILES = [
    GENERATE_PATH / "alg_dat.jsonl",
    GENERATE_PATH / "aligned_dataset.jsonl"
]

# Output file
OUTPUT_FILE = GENERATE_PATH / "combined_training_dataset.jsonl"


def combine_datasets(input_files, output_file):
    """
    Combine multiple JSONL files into one.
    
    Args:
        input_files: List of Path objects to input JSONL files
        output_file: Path object to output combined JSONL file
    """
    combined_data = []
    
    print("Combining datasets...")
    
    for input_file in input_files:
        if not input_file.exists():
            print(f"⚠ Warning: {input_file.name} not found, skipping...")
            continue
        
        print(f"  Reading {input_file.name}...")
        count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        combined_data.append(data)
                        count += 1
                    except json.JSONDecodeError as e:
                        print(f"    ⚠ Error parsing line: {e}")
        
        print(f"    Added {count} samples from {input_file.name}")
    
    # Write combined dataset
    print(f"\n  Writing combined dataset to {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Combined dataset saved!")
    print(f"  Total samples: {len(combined_data)}")
    print(f"  Output: {output_file}")
    
    return len(combined_data)


if __name__ == "__main__":
    total = combine_datasets(INPUT_FILES, OUTPUT_FILE)
    
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    print(f"Combined {len(INPUT_FILES)} datasets into {OUTPUT_FILE.name}")
    print(f"Total training samples: {total}")
