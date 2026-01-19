import json
from collections import defaultdict

"""
PURPOSE:
THIS CODE FIXES BUGS IN FINAL DATASET

"""



DATA_PATH = r"C:\Users\TheAiro\Desktop\Machine Learning Projects\Visuem\generate\t5_training_data_groq.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

print(data[0]["input"])

# Validation checks
print("\n=== Dataset Validation ===")
print(f"Total samples: {len(data)}")

# Check for missing required fields
required_fields = ['input', 'target']
missing_fields_count = 0
for idx, sample in enumerate(data):
    for field in required_fields:
        if field not in sample:
            print(f"❌ Sample {idx} missing '{field}' field")
            missing_fields_count += 1

print(f"Samples with missing fields: {missing_fields_count}")

# Check for empty values
empty_count = 0
for idx, sample in enumerate(data):
    if not sample.get('input') or not sample.get('target'):
        print(f"❌ Sample {idx} has empty input or target")
        empty_count += 1

print(f"Samples with empty values: {empty_count}")

# Check for duplicates
inputs = [sample.get('input') for sample in data]
duplicates = len(inputs) - len(set(inputs))
print(f"Duplicate samples: {duplicates}")

# Clean dataset: remove empty input or target instances
print("\n=== Cleaning Dataset ===")
cleaned_data = [sample for sample in data if sample.get('input') and sample.get('target')]
removed_count = len(data) - len(cleaned_data)
print(f"Removed {removed_count} samples with empty values")
print(f"Cleaned dataset size: {len(cleaned_data)}")

# Save cleaned data back to file
with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"✅ Updated {DATA_PATH} with cleaned data")