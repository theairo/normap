# Visuem

Ukrainian address normalization system. Takes messy natural language address queries and parses them into structured XML tags.

## What it does

Transforms unstructured Ukrainian address text into tagged format:

**Input:** `"біля макдональдсу на франка 5 кв 12"`  
**Output:** `<POI> макдональдсу </POI> <STREET> франка </STREET> <BUILD> 5 </BUILD> <ROOM> 12 </ROOM>`

The model handles:
- Different grammatical cases (nominative, locative, genitive)
- Colloquial expressions and prepositions
- Incomplete or partial addresses
- POI references (landmarks, stores, etc.)
- Delivery provider information

## Project Structure

```
Visuem/
├── data/                          # Raw and generated datasets
│   ├── raw/                       # Source data (streets, POIs, districts, providers)
│   │   ├── all_street_names_types.csv
│   │   ├── districts.csv
│   │   ├── poi.csv
│   │   ├── providers.csv
│   │   └── zhk.csv
│   ├── generated/                 # Training datasets
│   │   ├── dataset_final_version.jsonl  (52k+ samples)
│   │   ├── alg_dat.jsonl         # Algorithmically generated
│   │   └── aligned_dataset_clean.jsonl  # LLM-enhanced
│   └── testset/                   # Test data at 3 difficulty levels
│       ├── test_final_l1.txt     # Level 1: Simple addresses
│       ├── test_final_l2.txt     # Level 2: Medium complexity
│       └── test_final_l3.txt     # Level 3: Messy/colloquial
├── generate/                      # Dataset generation pipeline
│   ├── generate_algo.py          # Probabilistic data generator
│   ├── synthetic_groq_class.py   # Groq LLM integration
│   ├── synthetic_hf_class.py     # HuggingFace LLM integration
│   ├── generate_input.py         # Input text generation
│   ├── generate_target.py        # Target tag generation
│   └── llm_prompt.txt            # LLM prompt template
├── data_cleaning/                 # Data processing & validation
│   ├── combine_datasets.py       # Merge multiple datasets
│   ├── align_data.py             # Align inputs with targets
│   ├── validate_cleaned_dataset.py
│   ├── analyse_hallucinations.py # Detect LLM hallucinations
│   └── class_balance_analysis.ipynb
├── evaluate/                      # Model evaluation
│   ├── evaluate_model.py         # Metrics calculation
│   └── Ground_truth.txt          # Gold standard annotations
├── inference/                     # Production inference
│   ├── process_single.py         # Single query processing
│   └── process_csv.py            # Batch processing
└── requirements.txt               # Python dependencies
```

## Supported Tags

The model can extract these address components:

| Tag | Description | Example |
|-----|-------------|---------|
| `ZIP` | Postal code | 02000 |
| `CITY` | City/town | Київ, Львів |
| `OBLAST` | Region/oblast | Київська область |
| `DISTRICT` | District/neighborhood | Печерський район |
| `STREET` | Street name | вул. Хрещатик |
| `BUILD` | Building number | 15, 7А |
| `CORP` | Building corpus/block | корп. 2 |
| `ENTRANCE` | Entrance number | під'їзд 3 |
| `FLOOR` | Floor number | 5 поверх |
| `ROOM` | Flat/apartment/office | кв. 42, офіс 301 |
| `CODE` | Door/entrance code | код 1234 |
| `POI` | Point of interest | McDonalds, Ашан |
| `PROVIDER` | Delivery provider | Нова Пошта |
| `BRANCH_TYPE` | Branch type | Відділення, Поштомат |
| `BRANCH_ID` | Branch number | №44 |

## Setup

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional, for training)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/visuem.git
cd visuem

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (used in some scripts)
python -m spacy download en_core_web_sm
```

## Dataset Generation

The dataset uses two approaches:

### 1. Algorithmic Generation

Probabilistic generator with configurable tag probabilities:

```bash
python generate/generate_algo.py
```

**Features:**
- Independent probability distributions for each tag
- Conditional dependencies (e.g., flat requires building)
- Realistic street type distributions
- 52k+ samples generated

### 2. LLM-Enhanced Generation

Transforms canonical addresses into natural/messy queries:

```bash
python generate/synthetic_groq_class.py  # Using Groq
# or
python generate/synthetic_hf_class.py    # Using HuggingFace
```

**Transformations:**
- Grammatical case changes (Nominative → Locative/Genitive)
- POI anchoring with prepositions ("біля", "поруч з", "напроті")
- Noise injection (delivery instructions, visual descriptions)
- Colloquial expressions

### 3. Dataset Combination

```bash
python data_cleaning/combine_datasets.py
```

Merges multiple dataset sources with deduplication.

## Data Cleaning & Validation

### Alignment Verification

```bash
python data_cleaning/align_data.py
```

Ensures input-target pairs are correctly matched.

### Hallucination Detection

```bash
python data_cleaning/analyse_hallucinations.py
```

Detects when LLMs generate fake addresses not present in source data.

### Dataset Validation

```bash
python data_cleaning/validate_cleaned_dataset.py
```

Validates XML tag structure and data integrity.

## Model Evaluation

Run evaluation with:

```bash
python evaluate/evaluate_model.py \
    --ground-truth data/testset/test_final_l1.txt \
    --predictions predictions.jsonl \
    --output results.json
```

**Metrics:**
- **Sequence Accuracy**: Exact match of entire tagged output
- **Token-level Precision/Recall/F1**: Per-tag comparison
- **Tag-specific Performance**: Individual accuracy per tag type
- **Error Analysis**: Detailed mismatch reporting

### Test Set Levels

- **Level 1**: Clean, well-formed addresses
- **Level 2**: Medium complexity with variations
- **Level 3**: Messy, colloquial, incomplete queries

## Inference

### Single Query Processing

```python
import ctranslate2
import transformers

# Load model
tokenizer = transformers.AutoTokenizer.from_pretrained("model_path")
generator = ctranslate2.Translator("model_ct2_int8", device="cuda")

def normalize_query(text):
    input_text = f"normalize: {text.strip()}"
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    results = generator.translate_batch([source_tokens], beam_size=1)
    output_tokens = results[0].hypotheses[0]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))

# Usage
result = normalize_query("біля театру на театральній 5")
print(result)
# Output: <POI> театру </POI> <STREET> театральній </STREET> <BUILD> 5 </BUILD>
```

### Batch Processing

```bash
python inference/process_csv.py --input addresses.csv --output results.csv
```

## Data Format

### JSONL Format

```json
{
  "input": "біля макдональдсу на франка 5 кв 12",
  "target": "<POI> макдональдсу </POI> <STREET> франка </STREET> <BUILD> 5 </BUILD> <ROOM> 12 </ROOM>",
  "llm_mess": true
}
```

### Fields

- `input`: Raw user query (messy, natural language)
- `target`: XML-tagged structured output
- `llm_mess`: Boolean flag indicating LLM-generated messiness

## Model Architecture

Uses T5-based sequence-to-sequence models fine-tuned on Ukrainian addresses:

- **Input**: `"normalize: {messy_address}"`
- **Output**: XML-tagged structured address
- **Training**: ~52k samples with data augmentation
- **Inference**: CTranslate2 (INT8 quantization for speed)

## Dataset Statistics

- Total Samples: 52,144+
- Unique Cities: Thousands across Ukraine
- Street Types: 35k+ вулиця, 10k+ провулок, and more
- POIs: Universities, restaurants, shopping centers
- Providers: Нова Пошта, Rozetka, etc.

## Configuration

### Probability Settings (generate_algo.py)

```python
PROBS = {
    "has_city": 0.70,      # 70% samples include city
    "has_street": 0.85,    # 85% include street
    "has_building": 0.90,  # 90% include building number
    "has_room": 0.40,      # 40% include flat/office
    # ... more configurations
}
```

### Ignored Tags (evaluate_model.py)

```python
TAGS_TO_IGNORE = {"POI", "TYPE"}  # Tags excluded from evaluation
```

## Notes

This is specifically for Ukrainian addresses. The approach could probably work for other languages if you have the right data and adjust for grammatical rules.

The model training part isn't included here - just the data generation, cleaning, and evaluation scripts.
