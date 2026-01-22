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
    "has_poi": 0.1,         # 5% are Landmarks (McDonalds, Achan)

    # Street & Building: Independent binary rolls
    "has_street": 0.85,      # 85% have a street
    "has_building": 0.90,    # 90% have a building number
    "has_complex": 0.08,     # 8% mention residential complex name
    "has_corp": 0.05,        # 5% have a block/corpus

    # Sub-Units: Independent binary rolls
    "has_room": 0.40,        # 40% have a room number (flat/office)
    "has_entrance": 0.10,    # 10% mention entrance
    "has_floor": 0.10,       # 10% mention floor
    "has_code": 0.05,        # 5% mention door code

    # Provider details
    "has_branch_type": 0.60, # 60% of providers have branch type
    "has_branch_id": 0.60,   # 60% have branch ID
}

# Street type distribution
RARE_TYPES = ['проспект', 'бульвар', 'площа', 'набережна', 'узвіз']
COMMON_TYPES = ['вулиця', 'провулок']
RARE_TYPE_PROB = 0.40  # 40% chance to select from rare types

# Oblast/Region names
OBLASTS = ['Київська область', 'Львівська область', 'Харківська область', 
           'Одеська область', 'Дніпропетровська область', 'Запорізька область',
           'Полтавська область', 'Кіровоградська область', 'Хмельницька область']

# Residential complex names
COMPLEXES = ['ЖК Комфорт Таун', 'ЖК Преміум', 'ЖК Парк', 'ЖК Центр', 'ЖК Гавань',
             'ЖК Ліквувально', 'ЖК Львівський', 'ЖК Олімп']

# Provider branch types
BRANCH_TYPES = ['Відділення', 'Поштомат', 'Парцел', 'Пункт видачі', 'Філія', 'Каса']

# ==========================================
# DATA LOADING
# ==========================================

def load_data():
    """Load all necessary data sources"""
    
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
        streets_by_type = df_streets.groupby('type')['name'].apply(list).to_dict()
    except FileNotFoundError:
        print("Warning: kyiv_types_cleaned.csv not found. Using dummy data.")
    
    # C. Districts
    KYIV_DISTRICTS = ['Троєщина', 'Виноградар', 'Оболонь', 'Позняки', 
                      'Осокорки', 'Поділ', 'Печерськ', 'Солом\'янка']
    
    # D. Services & POIs
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
        'districts': KYIV_DISTRICTS,
        'providers': providers,
        'pois': pois
    }

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_random_street(streets_by_type):
    """Select a street with biased type distribution"""
    if random.random() < RARE_TYPE_PROB:
        avail = [t for t in RARE_TYPES if t in streets_by_type]
        chosen_type = random.choice(avail) if avail else "вулиця"
    else:
        avail = [t for t in COMMON_TYPES if t in streets_by_type]
        chosen_type = random.choice(avail) if avail else "вулиця"
    
    name = random.choice(streets_by_type.get(chosen_type, ["Шевченка"]))
    return name, chosen_type


def format_target_string(data):
    """Convert canonical dict to XML tag format"""
    parts = []
    
    # Order: ZIP → OBLAST → CITY → DISTRICT → STREET → TYPE → COMPLEX → BUILD → CORP → 
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
    if 'complex' in data: 
        parts.append(f"<COMPLEX> {data['complex']} </COMPLEX>")
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
        canonical['oblast'] = random.choice(OBLASTS)
    
    # City (70% probability)
    city_name = None
    if random.random() < PROBS["has_city"]:
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
    
    # Complex (8% probability)
    if random.random() < PROBS["has_complex"]:
        canonical['complex'] = random.choice(COMPLEXES)
    
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
        canonical['poi'] = poi['official_name']
    
    # ======================
    # BUILD INPUT STRING
    # ======================
    input_str = build_input_string(canonical, city_name, street_name, street_type)
    
    return input_str, canonical


def build_input_string(canonical, city_name, street_name, street_type):
    """
    Build natural Ukrainian input string from canonical data.
    Applies variations in formatting and word order.
    """
    parts = []
    
    # Variation templates
    city_format = random.choice([
        f"м. {city_name}",
        f"місто {city_name}",
        f"{city_name}",
        f"в {city_name}",
    ]) if city_name else ""
    
    # District
    if 'district' in canonical:
        district = canonical['district']
        parts.append(random.choice([
            f"район {district}",
            f"{district}",
            f"р-н {district}"
        ]))
    
    # Street
    if street_name and street_type:
        try:
            if street_type.lower() in street_name.lower():
                street_part = street_name
            else:
                # Different street formats
                street_part = random.choice([
                    f"{street_type} {street_name}",
                    f"{street_name} {street_type}",
                    f"{street_name}",  # Type omitted sometimes
                ])
            parts.append(street_part)
        except:
            print(street_name, street_type)
        # Anti-redundancy: if type is in name, don't duplicate
        
    
    # Building
    if 'build' in canonical:
        build_str = canonical['build']
        
        # Add corp if exists
        if 'corp' in canonical:
            build_str += random.choice([
                f" корп. {canonical['corp']}",
                f"/{canonical['corp']}",
                f"-{canonical['corp']}"
            ])
        
        parts.append(random.choice([
            f"буд. {build_str}",
            f"будинок {build_str}",
            f"{build_str}",
            f"№{build_str}"
        ]))
    
    # Complex (residential complex)
    if 'complex' in canonical:
        parts.append(canonical['complex'])
    
    # Room (apartment or office)
    if 'room' in canonical:
        # Randomly choose if it's apartment or office based on context
        if random.random() < 0.85:
            parts.append(random.choice([
                f"кв. {canonical['room']}",
                f"квартира {canonical['room']}",
                f"кв {canonical['room']}"
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
            prov_str += f" ({canonical['branch_type']})"
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

def generate_dataset(num_samples=1000, output_file="training_dataset.jsonl", variations_per_sample=5):
    """
    Generate complete dataset with multiple variations per canonical address.
    
    Args:
        num_samples: Number of unique canonical addresses
        output_file: Output JSONL file path
        variations_per_sample: Number of input variations per canonical
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
            
            input_str = build_input_string(canonical, city_name, street_name, street_type)
            
            dataset.append({
                "input": input_str,
                "target": target
            })
    
    # Save to file
    ROOT = Path(__file__).resolve().parents[1]
    output_path = ROOT / "generate" / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
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
        num_samples=1000,           # 200 unique addresses
        variations_per_sample=1,   # 5 variations each = 1000 total samples
        output_file="alg_dat.jsonl"
    )
    
    print("\n" + "="*50)
    print("Sample entries:")
    print("="*50)
    
