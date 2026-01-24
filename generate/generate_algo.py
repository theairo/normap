"""
Probabilistic Dataset Generator
================================
Generates address dataset samples based on tag probability distributions.

Each tag (city, street, flat, etc.) has an independent probability of appearing,
with some conditional dependencies (e.g., flat requires building).

Output Format: JSONL with {"input": "...", "target": "<TAG> ... </TAG>"}
"""

import pandas as pd
import random
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# PROBABILITY CONFIGURATION
# ==========================================
PROBS = {
    # Independent: Can appear alone
    "has_zip": 0.15,         # 15% have Zip Codes
    "has_city": 0.70,        # 70% have City
    "has_oblast": 0.10,      # 10% have Oblast (Region)
    "has_district": 0.20,    # 20% have District
    "has_provider": 0.15,    # 15% are Service points (NP, Rozetka)
    "has_poi": 0.2,         # 5% are Landmarks (McDonalds, Achan)

    # Street & Building: Independent binary rolls
    "has_street": 0.85,      # 85% have a street
    "has_building": 0.90,    # 90% have a building number
    "has_corp": 0.07,        # 5% have a block/corpus

    # Sub-Units: Independent binary rolls
    "has_room": 0.40,        # 40% have a room number (flat/office)
    "has_entrance": 0.10,    # 10% mention entrance
    "has_floor": 0.10,       # 10% mention floor
    "has_code": 0.08,        # 5% mention door code

    # Provider details
    "has_branch_type": 0.60, # 60% of providers have branch type
    "has_branch_id": 0.60,   # 60% have branch ID
}

# Street type distribution (bucketed with observed counts)
STREET_TYPE_COUNTS = {
    "вулиця": 35235,
    "провулок": 9980,
    "проїзд": 915,
    "в'їзд": 776,
    "тупик": 451,
    "площа": 392,
    "проспект": 327,
    "дорога": 207,
    "бульвар": 200,
    "алея": 140,
    "узвіз": 128,
    "шосе": 121,
    "лінія": 118,
    "майдан": 82,
    "набережна": 73,
    "траса": 26,
    "сквер": 10,
    "кільцева": 3,
    "магістраль": 2,
    "автодорога": 1,
}

COMMON_TYPES = ["вулиця", "провулок", "проїзд", "в'їзд", "тупик"]
RARE_TYPES = [
    "площа", "проспект", "дорога", "бульвар", "алея", "узвіз",
    "шосе", "лінія", "майдан", "набережна",
]
VERY_RARE_TYPES = ["траса", "сквер", "кільцева", "магістраль", "автодорога"]
STREET_BUCKET_WEIGHTS = {"common": 0.75, "rare": 0.2, "very_rare": 0.05}

# Street type abbreviations and variations
STREET_TYPE_VARIATIONS = {
    "вулиця": ["вулиця", "вул.", "вул"],
    "провулок": ["провулок", "пров.", "провул.", "пров"],
    "проїзд": ["проїзд", "пр-д"],
    "в'їзд": ["в'їзд"],
    "тупик": ["тупик"],
    "площа": ["площа", "пл."],
    "проспект": ["проспект", "просп.", "пр-т"],
    "дорога": ["дорога", "дор."],
    "бульвар": ["бульвар", "бульв.", "б-р"],
    "алея": ["алея", "ал."],
    "узвіз": ["узвіз", "узв."],
    "шосе": ["шосе", "ш."],
    "лінія": ["лінія"],
    "майдан": ["майдан", "майд."],
    "набережна": ["набережна", "наб.", "набер."],
    "траса": ["траса", "тр."],
    "сквер": ["сквер", "скв."],
    "кільцева": ["кільцева", "кільц."],
    "магістраль": ["магістраль", "маг."],
    "автодорога": ["автодорога", "а/д"],
}

# Fallback Oblast/Region names
FALLBACK_OBLASTS = [
    'Київська область', 'Львівська область', 'Харківська область',
    'Одеська область', 'Дніпропетровська область', 'Запорізька область',
    'Полтавська область', 'Кіровоградська область', 'Хмельницька область'
]

# Provider branch types
BRANCH_TYPES = ['Відділення', 'Поштомат', 'Пункт видачі', 'Філія', 'Каса']

# ==========================================
# DATA LOADING
# ==========================================

def load_data():
    """Load all necessary data sources"""
    
    cities = ['Київ']
    villages = ['Борщагівка']
    streets_by_type = {}
    fallback_districts = ['Троєщина', 'Виноградар', 'Оболонь', 'Позняки', 'Осокорки', 'Поділ', 'Печерськ', "Солом'янка"]
    districts = fallback_districts
    oblasts = FALLBACK_OBLASTS

    # A. Settlements
    try:
        df_places = pd.read_csv(ROOT / "data" / "ua-name-places.csv")
        cities = df_places[df_places['place'].isin(['city', 'town'])]['name'].dropna().tolist()
        villages = df_places[df_places['place'].isin(['village', 'hamlet'])]['name'].dropna().tolist()
    except FileNotFoundError:
        print("Warning: ua-name-places.csv not found. Using dummy data.")
    
    # B. Streets
    try:
        df_streets = pd.read_csv(ROOT / "data" / "all_street_names_types.csv")
        streets_by_type = df_streets.groupby('Type')['Name'].apply(list).to_dict()
    except FileNotFoundError:
        print("Warning: all_street_names_types.csv not found. Using dummy data.")

    # C. Districts
    try:
        df_districts = pd.read_csv(ROOT / "data" / "districts.csv")
        districts = df_districts['district'].dropna().unique().tolist()
        if not districts:
            districts = fallback_districts
    except FileNotFoundError:
        print("Warning: districts.csv not found. Using fallback districts.")

    # D. Oblasts
    try:
        df_oblasts = pd.read_csv(ROOT / "data" / "oblasti.csv", usecols=['Name'])
        oblasts = df_oblasts['Name'].dropna().tolist()
        if not oblasts:
            oblasts = FALLBACK_OBLASTS
    except FileNotFoundError:
        print("Warning: oblasti.csv not found. Using fallback oblasts.")
    
    # E. Services & POIs
    try:
        df_prov = pd.read_csv(ROOT / "data" / "providers.csv")
        df_poi = pd.read_csv(ROOT / "data" / "poi.csv", header=None, 
                            names=['category', 'official_name', 'tag_type', 'slang_variations'])
        
        providers = df_prov.to_dict('records')
        pois = df_poi.to_dict('records')
    except FileNotFoundError:
        print("Warning: providers.csv or poi.csv not found. Using dummy data.")
        providers = [
            {"official_name": "Нова Пошта", "tag_type": "PROVIDER"},
            {"official_name": "Укрпошта", "tag_type": "PROVIDER"}
        ]
        pois = [
            {"official_name": "McDonald's", "tag_type": "POI"},
            {"official_name": "Сільпо", "tag_type": "POI"}
        ]
    
    return {
        'cities': cities,
        'villages': villages,
        'streets_by_type': streets_by_type,
        'districts': districts,
        'oblasts': oblasts,
        'providers': providers,
        'pois': pois
    }

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _choose_street_bucket(available_types):
    """Select bucket based on configured weights and what is available."""
    bucket_map = {
        "common": [t for t in COMMON_TYPES if t in available_types],
        "rare": [t for t in RARE_TYPES if t in available_types],
        "very_rare": [t for t in VERY_RARE_TYPES if t in available_types],
    }

    buckets = [(name, types) for name, types in bucket_map.items() if types]
    if not buckets:
        return None, []

    names = [b[0] for b in buckets]
    weights = [STREET_BUCKET_WEIGHTS.get(b[0], 0.0) for b in buckets]
    chosen_bucket = random.choices(names, weights=weights, k=1)[0]
    return chosen_bucket, dict(buckets)[chosen_bucket]


def get_random_street(streets_by_type):
    """Select a street type using buckets and weighted counts."""
    available_types = {k: v for k, v in streets_by_type.items() if v}
    bucket, bucket_types = _choose_street_bucket(available_types)

    if bucket_types:
        type_weights = [STREET_TYPE_COUNTS.get(t, 1) for t in bucket_types]
        chosen_type = random.choices(bucket_types, weights=type_weights, k=1)[0]
    else:
        # Fallback: use any available type or default to вулиця
        fallback_types = list(available_types.keys()) or ["вулиця"]
        chosen_type = random.choice(fallback_types)
    
    name = random.choice(available_types.get(chosen_type, ["Шевченка"]))
    return name, chosen_type


def format_target_string(data):
    """Convert canonical dict to XML tag format"""
    parts = []
    
    # Order: ZIP → OBLAST → CITY → DISTRICT → STREET → TYPE → BUILD → CORP → 
    # ENTRANCE → FLOOR → ROOM → CODE → PROVIDER → BRANCH_TYPE → BRANCH_ID → POI
    
    if 'zip' in data: 
        parts.append(f"<ZIP> {data['zip']} </ZIP>")
    if 'oblast' in data: 
        parts.append(f"<OBLAST> {data['oblast']} </OBLAST>")
    if 'city' in data: 
        parts.append(f"<CITY> {data['city']} </CITY>")
    if 'district' in data: 
        parts.append(f"<DISTRICT> {data['district']} </DISTRICT>")
    if 'street' in data: 
        parts.append(f"<STREET> {data['street']} </STREET>")
    if 'type' in data: 
        parts.append(f"<TYPE> {data['type']} </TYPE>")
    if 'build' in data: 
        parts.append(f"<BUILD> {data['build']} </BUILD>")
    if 'corp' in data: 
        parts.append(f"<CORP> {data['corp']} </CORP>")
    if 'entrance' in data: 
        parts.append(f"<ENTRANCE> {data['entrance']} </ENTRANCE>")
    if 'floor' in data: 
        parts.append(f"<FLOOR> {data['floor']} </FLOOR>")
    if 'room' in data: 
        parts.append(f"<ROOM> {data['room']} </ROOM>")
    if 'code' in data: 
        parts.append(f"<CODE> {data['code']} </CODE>")
    if 'provider' in data: 
        parts.append(f"<PROVIDER> {data['provider']} </PROVIDER>")
    if 'branch_type' in data: 
        parts.append(f"<BRANCH_TYPE> {data['branch_type']} </BRANCH_TYPE>")
    if 'branch_id' in data: 
        parts.append(f"<BRANCH_ID> {data['branch_id']} </BRANCH_ID>")
    if 'poi' in data: 
        parts.append(f"<POI> {data['poi']} </POI>")
    
    return " ".join(parts)


def generate_building_number():
    """Generate realistic building number"""
    building = str(random.randint(1, 150))
    # Add suffix to some buildings
    if random.random() < 0.1:
        building += random.choice(["А", "Б", "В", "/1", "/2"])
    return building


def generate_zip_code():
    """Generate realistic Ukrainian postal code"""
    return f"{random.randint(10, 99)}{random.randint(100, 999):03d}"


# ==========================================
# MAIN GENERATION LOGIC
# ==========================================
    
def generate_sample(data_sources):
    """
    Generate a single address sample using BINARY probabilities.
    
    ALL tags roll independently - each has its own probability.
    No hard dependencies, purely probabilistic.
    
    Returns: (input_string, canonical_dict)
    """
    canonical = {}
    
    # ======================
    # INDEPENDENT BINARY ROLLS
    # ======================
    
    # ZIP code (15% probability)
    if random.random() < PROBS["has_zip"]:
        canonical['zip'] = generate_zip_code()
    
    # Oblast (10% probability)
    if random.random() < PROBS["has_oblast"]:
        canonical['oblast'] = random.choice(data_sources['oblasts'])
    
    # City (70% probability) - includes villages
    city_name = None
    is_village = False
    if random.random() < PROBS["has_city"]:
        # 20% chance to pick a village instead of a city
        if random.random() < 0.2:
            city_name = random.choice(data_sources['villages'])
            is_village = True
        else:
            city_name = random.choice(data_sources['cities'])
        canonical['city'] = city_name
    
    # District (20% probability)
    if random.random() < PROBS["has_district"]:
        canonical['district'] = random.choice(data_sources['districts'])
    
    # Street (85% probability)
    street_name = None
    street_type = None
    if random.random() < PROBS["has_street"]:
        street_name, street_type = get_random_street(data_sources['streets_by_type'])
        canonical['street'] = street_name
        canonical['type'] = street_type
    
    # Building (90% probability) - ONLY if Street exists
    if random.random() < PROBS["has_building"] and 'street' in canonical:
        canonical['build'] = generate_building_number()
    
    # Corp (5% probability)
    if random.random() < PROBS["has_corp"]:
        canonical['corp'] = random.choice(['А', 'Б', '1', '2'])
    
    # Room (40% probability) - ONLY if Building exists
    if random.random() < PROBS["has_room"] and 'build' in canonical:
        canonical['room'] = str(random.randint(1, 200))
    
    # Entrance (10% probability) - ONLY if Building exists
    if random.random() < PROBS["has_entrance"] and 'build' in canonical:
        canonical['entrance'] = str(random.randint(1, 5))
    
    # Floor (10% probability) - ONLY if Building exists
    if random.random() < PROBS["has_floor"] and 'build' in canonical:
        canonical['floor'] = str(random.randint(1, 20))
    
    # Code (5% probability) - ONLY if Building exists
    if random.random() < PROBS["has_code"] and 'build' in canonical:
        canonical['code'] = str(random.randint(100, 9999))
    
    # Provider (15% probability)
    if random.random() < PROBS["has_provider"]:
        provider = random.choice(data_sources['providers'])
        canonical['provider'] = provider['official_name']
        
        # Branch Type (60% of providers) - ONLY if Provider exists
        if random.random() < PROBS["has_branch_type"]:
            canonical['branch_type'] = random.choice(BRANCH_TYPES)
        
        # Branch ID (60% of providers) - ONLY if Provider exists
        if random.random() < PROBS["has_branch_id"]:
            canonical['branch_id'] = str(random.randint(1, 300))
    
    # POI (5% probability)
    if random.random() < PROBS["has_poi"]:
        poi = random.choice(data_sources['pois'])
        # Use official name or slang variation
        if poi.get('slang_variations') and random.random() < 0.6:
            # Parse slang variations (pipe-separated)
            slang_list = [s.strip() for s in str(poi['slang_variations']).split('|')]
            canonical['poi'] = random.choice(slang_list)
        else:
            canonical['poi'] = poi['official_name']
    
    # ======================
    # BUILD INPUT STRING
    # ======================
    input_str = build_input_string(canonical, city_name, street_name, street_type, is_village)
    
    return input_str, canonical


def build_input_string(canonical, city_name, street_name, street_type, is_village=False):
    """
    Build natural Ukrainian input string from canonical data.
    Applies variations in formatting and word order.
    """
    parts = []
    
    # Oblast
    if 'oblast' in canonical:
        oblast = canonical['oblast']
        parts.append(random.choice([
            f"обл. {oblast}",
            f"область {oblast}",
            oblast,
        ]))
    
    # Variation templates for city/village
    city_format = ""
    if city_name:
        if is_village:
            # Village formatting with various prefixes
            city_format = random.choice([
                f"село {city_name}",
                f"с. {city_name}",
                f"селище {city_name}",
                f"смт {city_name}",
                f"смт. {city_name}",
                f"селище міського типу {city_name}",
                f"{city_name}",  # Sometimes no prefix
            ])
        else:
            # City formatting
            city_format = random.choice([
                f"м. {city_name}",
                f"місто {city_name}",
                f"{city_name}",
            ])

    # City/Village (add before shuffling so it appears in input)
    if city_format:
        parts.append(city_format)
    
    # District
    if 'district' in canonical:
        district = canonical['district']
        parts.append(random.choice([
            f"район {district}",
            f"{district}",
            f"р-н {district}"
        ]))
    
    # Street and building (building must follow street)
    street_building_part = None
    if street_name and street_type:
        try:
            if street_type.lower() in street_name.lower():
                street_part = street_name
            else:
                # Get random variation of street type
                type_variations = STREET_TYPE_VARIATIONS.get(street_type, [street_type])
                street_type_var = random.choice(type_variations)
                
                street_part = random.choice([
                    f"{street_type_var} {street_name}",
                    f"{street_name} {street_type_var}",
                    f"{street_name}",  # Type omitted sometimes
                ])
        except Exception:
            street_part = street_name

        if 'build' in canonical:
            build_str = canonical['build']
            if 'corp' in canonical:
                build_str += random.choice([
                    f" корп. {canonical['corp']}",
                    f"/{canonical['corp']}",
                    f"-{canonical['corp']}"
                ])
            building_form = random.choice([
                f"буд. {build_str}",
                f"будинок {build_str}",
                f"{build_str}",
                f"№{build_str}"
            ])
            sep = ", " if random.random() < 0.7 else " "
            street_building_part = f"{street_part}{sep}{building_form}"
        else:
            street_building_part = street_part

    if street_building_part:
        parts.append(street_building_part)
    
    # Room (apartment or office)
    if 'room' in canonical:
        # Randomly choose if it's apartment or office based on context
        if random.random() < 0.6:
            parts.append(random.choice([
                f"кв. {canonical['room']}",
                f"квартира {canonical['room']}",
                f"кв {canonical['room']}"
            ]))
        elif random.random() < 0.8:
            parts.append(random.choice([
                f"каб. {canonical['room']}",
                f"кабінет {canonical['room']}",
            ]))
        else:
            parts.append(random.choice([
                f"оф. {canonical['room']}",
                f"офіс {canonical['room']}",
            ]))
    
    # Entrance
    if 'entrance' in canonical:
        parts.append(random.choice([
            f"під'їзд {canonical['entrance']}",
            f"п. {canonical['entrance']}",
        ]))
    
    # Floor
    if 'floor' in canonical:
        parts.append(random.choice([
            f"поверх {canonical['floor']}",
            f"{canonical['floor']} поверх",
        ]))
    
    # Door code
    if 'code' in canonical:
        parts.append(random.choice([
            f"код {canonical['code']}",
            f"домофон {canonical['code']}",
        ]))
    
    # Provider
    if 'provider' in canonical:
        prov_str = canonical['provider']
        if 'branch_type' in canonical:
            prov_str += f" {canonical['branch_type']}"
        if 'branch_id' in canonical:
            prov_str += f" №{canonical['branch_id']}"
        parts.append(prov_str)
    
    # POI
    if 'poi' in canonical:
        parts.append(canonical['poi'])

    # ZIP
    if 'zip' in canonical:
        parts.append(canonical['zip'])
    
    # Randomize order of parts (but keep related items together)
    random.shuffle(parts)
    
    # Join with random separators (comma or space for each split)
    if not parts:
        return ""
    
    result = parts[0]
    for part in parts[1:]:
        # 70% comma+space, 30% just space
        separator = ", " if random.random() < 0.7 else " "
        result += separator + part
    
    return result


# ==========================================
# DATASET GENERATION
# ==========================================

def generate_dataset(num_samples=1000, output_file="training_dataset.jsonl", variations_per_sample=5, generate_for_llm=False):
    """
    Generate complete dataset with multiple variations per canonical address.
    
    Args:
        num_samples: Number of unique canonical addresses
        output_file: Output JSONL file path
        variations_per_sample: Number of input variations per canonical
        generate_for_llm: If True, saves only inputs (no tags) to gen_for_llm.jsonl
    """

    data_sources = load_data()
    dataset = []
    
    print(f"Generating {num_samples} samples with {variations_per_sample} variations each...")
    
    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
        
        # Generate one canonical sample
        _, canonical = generate_sample(data_sources)
        target = format_target_string(canonical)
        
        # Generate multiple input variations
        for _ in range(variations_per_sample):
            # Rebuild input with random variations
            city_name = canonical.get('city')
            street_name = canonical.get('street')
            street_type = canonical.get('type')
            # Check if it's a village by seeing if it exists in villages list
            is_village = city_name in data_sources['villages'] if city_name else False
            
            input_str = build_input_string(canonical, city_name, street_name, street_type, is_village)
            
            dataset.append({
                "input": input_str,
                "target": target
            })
    
    # Save to file
    ROOT = Path(__file__).resolve().parents[1]
    
    if generate_for_llm:
        # Save inputs to gen_for_llm.jsonl (one per line)
        input_path = ROOT / "generate" / "gen_for_llm.jsonl"
        with open(input_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                if item["input"].strip():
                    f.write(item["input"] + '\n')
        
        # Save targets to gen_for_llm_targets.jsonl (one per line)
        target_path = ROOT / "generate" / "gen_for_llm_targets.jsonl"
        with open(target_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                if item["target"].strip():
                    f.write(item["target"] + '\n')
        
        output_path = input_path
        print(f"\n✓ Inputs saved to {input_path}")
        print(f"✓ Targets saved to {target_path}")
    else:
        # Save full JSONL format with input and target
        output_path = ROOT / "generate" / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Skip entries with empty input or target
                if item["input"].strip() and item["target"].strip():
                    # Add llm_mess flag
                    item_with_flag = {
                        "input": item["input"],
                        "target": item["target"],
                        "llm_mess": False
                    }
                    f.write(json.dumps(item_with_flag, ensure_ascii=False) + '\n')
    
    if not generate_for_llm:
        print(f"\n✓ Dataset saved to {output_path}")
    
    print(f"  Total samples: {len(dataset)}")
    print(f"  Unique addresses: {num_samples}")
    
    return dataset


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Generate dataset matching final_training_dataset format
    generate_dataset(
        num_samples=50000,           # 200 unique addresses
        variations_per_sample=1,   # 5 variations each = 1000 total samples
        output_file="alg_dat.jsonl",
        generate_for_llm=False       # Set to True to generate plain input list
    )
    
    print("\n" + "="*50)
    print("Sample entries:")
    print("="*50)
    
