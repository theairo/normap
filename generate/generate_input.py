import os
import sys
import json
import math
import time
import re
from pathlib import Path
from dotenv import load_dotenv


# Resolve project root
ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = ROOT / "generate"
if str(GEN_PATH) not in sys.path: sys.path.insert(0, str(GEN_PATH))
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from generate.synthetic_groq_class import GroqDataFactory
from generate.generate_target import generate_single_sample
#from generate.synthetic_hf_class import HuggingFaceDataFactory

load_dotenv(r"C:\Users\TheAiro\Desktop\Machine Learning Projects\Visuem\generate\.env.local")

# CONFIGURATION
PROMPT_TEMPLATE_PATH = ROOT / "data" / "llm_prompt.txt"
TEMP_CHUNK_OUTPUT = ROOT / "generate" / "temp_chunk_raw.txt"
FINAL_DATASET_FILE = ROOT / "generate" / "final_training_dataset.jsonl"

TARGET_PROMPTS = 4000
PROMPT_CHUNK_SIZE = 2 

PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

def prompt_builder(batch):
    item = batch[0]
    prompt = (
        PROMPT_TEMPLATE
        .replace("{{scenario}}", item.get("scenario", ""))
        .replace("{{clean_input}}", item.get("clean_input", ""))
        .replace("{{canonical}}", item.get("canonical", ""))
    )
    return prompt

def clean_json_response(text):
    """
    Cleans LLM response to ensure valid JSON parsing.
    Removes markdown ```json ... ``` and extra whitespace.
    """
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    
    # Attempt to find the list bracket structure [ ... ]
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def count_existing_progress():
    if not FINAL_DATASET_FILE.exists(): return 0
    line_count = 0
    with open(FINAL_DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): line_count += 1
    return line_count // 5

def main():
    processed_prompts = count_existing_progress()
    print(f"RESUME STATUS: Found {processed_prompts * 5} samples already generated.")
    
    if processed_prompts >= TARGET_PROMPTS:
        print("Target reached!")
        return

    remaining_prompts = TARGET_PROMPTS - processed_prompts
    total_chunks = math.ceil(remaining_prompts / PROMPT_CHUNK_SIZE)
    
    print(f"Remaining: {remaining_prompts} prompts. Processing in {total_chunks} chunks...")

    # factory = GroqDataFactory(
    #     api_key=os.getenv("HF_TOKEN"),
    #     model_name="meta-llama/Llama-3.3-70B-Instruct" 
    # )

    factory = GroqDataFactory(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b",
    )

    for chunk_idx in range(total_chunks):
        current_batch_size = min(PROMPT_CHUNK_SIZE, remaining_prompts)
        print(f"\n--- Processing Chunk {chunk_idx + 1}/{total_chunks} ({current_batch_size} prompts) ---")
        
        # A. Generate Seeds
        chunk_seeds = [generate_single_sample() for _ in range(current_batch_size)]
        
        # B. Clear temp file
        if TEMP_CHUNK_OUTPUT.exists(): os.remove(TEMP_CHUNK_OUTPUT)

        # C. Run LLM
        try:

            factory.run(
                inputs=chunk_seeds,
                prompt_builder=prompt_builder,
                output_file=str(TEMP_CHUNK_OUTPUT),
                batch_size=1,
                target_count=current_batch_size,
                sleep_between_batches=15,
                inputs_only=True,
                plain_inputs_only=True, # This will write one JSON string per line
            )

        except Exception as e:
            print(f"Error: {e}. Retrying chunk...")
            time.sleep(10)
            continue

        # D. Read Output
        if not TEMP_CHUNK_OUTPUT.exists(): continue
        
        with open(TEMP_CHUNK_OUTPUT, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        # E. THE FIX: 1-to-1 MAPPING
        # Since we asked for JSON list, we expect 1 line per prompt.
        
        new_samples = []
        
        # We zip strictly: 1 Seed <-> 1 Output Line
        # Even if the LLM output nonsense, it won't break the alignment of future rows.
        for seed, raw_line in zip(chunk_seeds, raw_lines):
            try:
                # Clean and Parse JSON
                cleaned_line = clean_json_response(raw_line)
                variations = json.loads(cleaned_line)
                
                # Check if it's actually a list
                if isinstance(variations, list):
                    for variation in variations:
                        new_samples.append({
                            "input": str(variation), # Ensure string
                            "target": seed['canonical'] # The link is preserved!
                        })
                else:
                    print(f"Warning: Output was valid JSON but not a list: {cleaned_line}")

            except json.JSONDecodeError:
                print(f"Skipping bad JSON line: {raw_line[:50]}...")
                continue

        # F. Append to Final File
        if new_samples:
            with open(FINAL_DATASET_FILE, "a", encoding="utf-8") as f:
                for sample in new_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            print(f"Saved {len(new_samples)} samples. Progress: {(processed_prompts + current_batch_size) * 5}")
            processed_prompts += current_batch_size
            remaining_prompts -= current_batch_size
        else:
            print("Warning: 0 valid samples in this chunk.")

    print(f"\nDONE! Dataset: {FINAL_DATASET_FILE}")

if __name__ == "__main__":
    main()