import ctranslate2
import transformers
import csv
import os
import time
import torch
from math import ceil

# --- CONFIGURATION --- # CHANGE THESE! 
TOKENIZER_PATH = "/kaggle/working/geo_normalizer_model"
CONVERTED_MODEL_PATH = "/kaggle/working/geo_model_ct2_int8"
OUTPUT_DIR = "/kaggle/working/"

# 📋 LIST OF CSV DATASETS TO PROCESS
# Ensure your CSV has a header or adjust the loading function
TEST_FILES = [
    ("LAYER 3 CSV", "/kaggle/input/your_dataset/test_data.csv")
]

BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_COLUMN_NAME = "input"  # Change this to match your CSV header

# --- 1. INITIALIZE MODEL ---
print(f"Loading Model & Tokenizer on {DEVICE}...")
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)
generator = ctranslate2.Translator(CONVERTED_MODEL_PATH, device=DEVICE)

# --- HELPER FUNCTIONS ---
def normalize_batch(texts):
    """Tokenizes and translates a batch of texts."""
    inputs = [f"normalize: {str(t).strip()}" for t in texts]
    source_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(i)) for i in inputs]
    
    results = generator.translate_batch(
        source_tokens,
        max_batch_size=BATCH_SIZE,
        beam_size=1,
        max_decoding_length=256
    )
    
    decoded = []
    for result in results:
        output_tokens = result.hypotheses[0]
        text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        text = text.replace("normalize:", "").strip()
        decoded.append(text)
    return decoded

def load_csv_data(path, column_name):
    data = []
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row[column_name])
    return data

# --- MAIN GENERATION LOOP ---
for dataset_name, file_path in TEST_FILES:
    print(f"\n" + "#"*60)
    print(f"PROCESSING CSV: {dataset_name}")
    print("#"*60)
    
    # 1. Load Data
    inputs_list = load_csv_data(file_path, INPUT_COLUMN_NAME)
    if not inputs_list: continue
    print(f"Loaded {len(inputs_list)} samples.")

    # 2. Prepare Output CSV
    input_filename = os.path.basename(file_path)
    output_filename = f"preds_{input_filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"💾 Saving predictions to: {output_path}")

    # 3. Batch Inference
    start_time = time.time()
    num_batches = ceil(len(inputs_list) / BATCH_SIZE)
    
    with open(output_path, "w", encoding="utf-8", newline='') as out_f:
        writer = csv.writer(out_f)
        # Write Header
        writer.writerow(["input", "prediction"])
        
        for i in range(num_batches):
            batch_inputs = inputs_list[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            
            # Generate
            preds = normalize_batch(batch_inputs)
            
            # Save to CSV immediately
            for inp, pred in zip(batch_inputs, preds):
                writer.writerow([inp, pred])
            
            if i % 5 == 0:
                print(f"   ... processed batch {i+1}/{num_batches}")

    duration = time.time() - start_time
    print(f"✅ Done! Speed: {len(inputs_list)/duration:.2f} samples/sec")