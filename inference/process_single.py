import ctranslate2
import transformers
import torch

# --- CONFIGURATION ---
TOKENIZER_PATH = "/kaggle/working/geo_normalizer_model"
CONVERTED_MODEL_PATH = "/kaggle/working/geo_model_ct2_int8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. INITIALIZE MODEL ---
print(f"Loading Model & Tokenizer on {DEVICE}...")
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)
generator = ctranslate2.Translator(CONVERTED_MODEL_PATH, device=DEVICE)

# --- INFERENCE FUNCTION ---
def normalize_query(text):
    """Tokenizes and translates a single string."""
    # Add prefix required by the T5 model
    input_text = f"normalize: {text.strip()}"
    
    # Encode and convert to tokens
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    
    # Generate translation
    results = generator.translate_batch(
        [source_tokens], 
        beam_size=1, 
        max_decoding_length=256
    )
    
    # Decode result
    output_tokens = results[0].hypotheses[0]
    prediction = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
    
    # Clean up output
    return prediction.replace("normalize:", "").strip()

# --- EXAMPLE USAGE ---
query = "напроти театру на театральній 5"
result = normalize_query(query)

print("-" * 30)
print(f"Input:  {query}")
print(f"Output: {result}")
print("-" * 30)