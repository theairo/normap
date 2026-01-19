import pandas as pd
import random
import numpy as np
import json

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================

# A. Settlements
try:
    df_places = pd.read_csv("./data/ua-name-places.csv")
    cities = df_places[df_places['place'].isin(['city', 'town'])]['name'].tolist()
    villages = df_places[df_places['place'].isin(['village', 'hamlet'])]['name'].tolist()
except FileNotFoundError:
    print("Warning: ua-name-places.csv not found. Using dummy data.")
    cities = ["Київ", "Львів", "Харків", "Одеса", "Дніпро"]
    villages = ["Гатне", "Щасливе", "Гора"]

# B. Streets
try:
    df_streets = pd.read_csv("./data/kyiv_types_cleaned.csv")
    streets_by_type = df_streets.groupby('name_type')['name'].apply(list).to_dict()
except FileNotFoundError:
    print("Warning: kyiv_types_cleaned.csv not found. Using dummy data.")
    streets_by_type = {"вулиця": ["Франка", "Шевченка"], "проспект": ["Науки", "Перемоги"]}

RARE_TYPES = ['проспект', 'бульвар', 'площа', 'набережна', 'узвіз']
COMMON_TYPES = ['вулиця', 'провулок']

# C. Districts
KYIV_DISTRICTS = ['Троєщина', 'Виноградар', 'Оболонь', 'Позняки', 'Осокорки', 'Поділ', 'Печерськ']

# D. Services & POIs
try:
    df_prov = pd.read_csv("./data/providers.csv")
    df_poi = pd.read_csv("./data/poi.csv", header=None, names=['category', 'official_name', 'tag_type', 'slang_variations'])
    df_services = pd.concat([df_prov, df_poi])
    logistics_providers = df_services[df_services['category'] == 'logistics']
    poi_providers = df_services[df_services['category'] != 'logistics']
except FileNotFoundError:
    print("Warning: providers.csv or poi.csv not found.")
    logistics_providers = pd.DataFrame()
    poi_providers = pd.DataFrame()

# ==========================================
# 2. HELPER: DICT TO XML CONVERTER (NEW!)
# ==========================================
def format_target_string(data):
    """Converts the canonical dict into the T5 Training String format."""
    parts = []
    
    # Order matters slightly for consistency, though T5 is robust.
    # 1. Location High Level
    if 'city' in data: parts.append(f"<CITY> {data['city']} </CITY>")
    if 'district' in data: parts.append(f"<DISTRICT> {data['district']} </DISTRICT>")
    
    # 2. Street Level
    if 'street' in data: parts.append(f"<STREET> {data['street']} </STREET>")
    if 'type' in data: parts.append(f"<TYPE> {data['type']} </TYPE>")
    
    # 3. Building Level
    if 'build' in data: parts.append(f"<BUILD> {data['build']} </BUILD>")
    if 'entrance' in data: parts.append(f"<ENTRANCE> {data['entrance']} </ENTRANCE>")
    if 'code' in data: parts.append(f"<CODE> {data['code']} </CODE>")
    if 'flat' in data: parts.append(f"<FLAT> {data['flat']} </FLAT>")
    
    # 4. Service/POI Level
    if 'provider' in data: parts.append(f"<PROVIDER> {data['provider']} </PROVIDER>")
    if 'poi' in data: parts.append(f"<POI> {data['poi']} </POI>")
    if 'poi_type' in data: parts.append(f"<TYPE> {data['poi_type']} </TYPE>")
    
    if 'branch_id' in data: parts.append(f"<BRANCH_ID> {data['branch_id']} </BRANCH_ID>")
    if 'branch_type' in data: parts.append(f"<BRANCH_TYPE> {data['branch_type']} </BRANCH_TYPE>")

    return " ".join(parts)

# ==========================================
# 3. GENERATION LOGIC
# ==========================================

def get_random_street():
    if random.random() < 0.40: 
        avail = [t for t in RARE_TYPES if t in streets_by_type]
        chosen_type = random.choice(avail) if avail else "вулиця"
    else:
        chosen_type = random.choice(COMMON_TYPES)
        
    name = random.choice(streets_by_type.get(chosen_type, ["Шевченка"]))
    return name, chosen_type

def add_level4_details():
    rand = random.random()
    building = str(random.randint(1, 150))
    if random.random() < 0.1: building += random.choice(["А", "Б", "/1"])

    if rand < 0.50:  # Building only
        return building, {"build": building}
    elif rand < 0.70:  # Flat
        flat = random.randint(1, 200)
        return f"{building}, кв. {flat}", {"build": building, "flat": flat}
    elif rand < 0.90:  # Full detail
        ent = random.randint(1, 5)
        code = random.randint(100, 9999)
        return f"{building}, під'їзд {ent}, код {code}", {"build": building, "entrance": ent, "code": code}
    else:  # No number
        return "", {}

def gen_urban_scenario():
    """Standard City Scenario"""
    city = random.choice(cities)
    street, s_type = get_random_street()
    
    # ANTI-REDUNDANCY FIX (Implemented in Python)
    # If the street name implies the type, we clean the input string
    if s_type.lower() in street.lower():
        clean_street_input = street
    else:
        clean_street_input = f"{s_type} {street}"
        
    details_input, details = add_level4_details()

    canonical = {"city": city, "street": street, "type": s_type}
    canonical.update(details)

    if details_input:
        input_str = f"м. {city}, {clean_street_input}, {details_input}"
    else:
        input_str = f"м. {city}, {clean_street_input}"
        
    return input_str, canonical

def gen_hyperlocal_scenario():
    """Implicit City Scenario (Kyiv assumed)"""
    street, s_type = get_random_street()
    
    if s_type.lower() in street.lower():
        clean_street_input = street
    else:
        clean_street_input = f"{s_type} {street}"
        
    details_input, details = add_level4_details()

    # Canonical HAS city, Input DOES NOT
    canonical = {"city": "Київ", "street": street, "type": s_type}
    canonical.update(details)

    input_str = f"{clean_street_input}{', ' + details_input if details_input else ''}"
    return input_str, canonical

def gen_district_scenario():
    """District Scenario"""
    district = random.choice(KYIV_DISTRICTS)
    street, s_type = get_random_street()
    
    if s_type.lower() in street.lower():
        clean_street_input = street
    else:
        clean_street_input = f"{s_type} {street}"
        
    details_input, details = add_level4_details()

    canonical = {"city": "Київ", "district": district, "street": street, "type": s_type}
    canonical.update(details)

    input_str = f"{district}, {clean_street_input}{', ' + details_input if details_input else ''}"
    return input_str, canonical

def gen_village_scenario():
    """Village Scenario"""
    vill = random.choice(villages)
    street, s_type = get_random_street()
    
    if s_type.lower() in street.lower():
        clean_street_input = street
    else:
        clean_street_input = f"{s_type} {street}"

    # Fixed key consistency: 'street_type' -> 'type'
    canonical = {"city": vill, "type": "село", "street": street, "type": s_type}
    input_str = f"с. {vill}, {clean_street_input}"
    return input_str, canonical

def gen_service_scenario():
    """Nova Poshta"""
    if logistics_providers.empty: return gen_urban_scenario()
    
    row = logistics_providers.sample(1).iloc[0]
    slang = random.choice(str(row['slang_variations']).split('|'))
    branch = random.randint(1, 150)
    
    # Added City Context to be safe
    city = random.choice(cities)

    canonical = {"city": city, "provider": row.get('official_name', ''), "branch_id": branch}
    input_str = f"м. {city}, {slang} {branch}"
    return input_str, canonical

def gen_poi_scenario():
    """Malls / POI"""
    if poi_providers.empty: return gen_urban_scenario()

    row = poi_providers.sample(1).iloc[0]
    slang = random.choice(str(row['slang_variations']).split('|'))
    city = random.choice(cities)

    canonical = {"city": city, "poi": row.get('official_name', '')}
    # Map 'tag_type' to 'poi_type'
    if 'tag_type' in row.index and pd.notna(row['tag_type']) and row['tag_type'] != 'POI':
        canonical["poi_type"] = row['tag_type']

    input_str = f"м. {city}, біля {slang}"
    return input_str, canonical

def gen_agnostic_urban_scenario():
    """Generates Street + Details WITHOUT A CITY.
    Target: <STREET> ... <BUILD> ... (No City Tag)"""
    
    street, s_type = get_random_street()
    
    # Anti-Redundancy Logic
    if s_type.lower() in street.lower():
        clean_street_input = street
    else:
        clean_street_input = f"{s_type} {street}"
        
    details_input, details = add_level4_details()

    # CANONICAL HAS NO CITY
    canonical = {"street": street, "type": s_type}
    canonical.update(details)

    if details_input:
        input_str = f"{clean_street_input}, {details_input}"
    else:
        input_str = clean_street_input
        
    return input_str, canonical

def gen_agnostic_service_scenario():
    """Generates Service Point WITHOUT A CITY.
    Target: <PROVIDER> ... <BRANCH_ID> ... (No City Tag)"""
    
    if logistics_providers.empty: return gen_agnostic_urban_scenario()
    
    row = logistics_providers.sample(1).iloc[0]
    slang = random.choice(str(row['slang_variations']).split('|'))
    branch = random.randint(1, 150)
    
    # CANONICAL HAS NO CITY
    canonical = {"provider": row.get('official_name', ''), "branch_id": branch}
    input_str = f"{slang} {branch}"
    
    return input_str, canonical

# ==========================================
# 4. MAIN GENERATOR (The Balanced Mix)
# ==========================================
def generate_single_sample(seed: int | None = None):
    if seed is not None: random.seed(seed)

    scenario_roll = random.random()

    # 1. FULL URBAN (40%) - The "Perfect" Address
    if scenario_roll < 0.40:
        clean, canon_dict = gen_urban_scenario()
        scenario_type = "urban_full"
        
    # 2. AGNOSTIC URBAN (20%) - "Sadova 5" (NO CITY) <--- CRITICAL NEW BUCKET
    elif scenario_roll < 0.60:
        clean, canon_dict = gen_agnostic_urban_scenario()
        scenario_type = "urban_agnostic"
        
    # 3. AGNOSTIC SERVICE (15%) - "Nova Poshta 5" (NO CITY) <--- CRITICAL NEW BUCKET
    elif scenario_roll < 0.75:
        clean, canon_dict = gen_agnostic_service_scenario()
        scenario_type = "service_agnostic"

    # 4. DISTRICT CONTEXT (10%) - "Troieshchyna, Zakrevskoho" (Implicit Kyiv)
    # Keeping this is okay because Districts imply a city, unlike generic streets.
    elif scenario_roll < 0.85:
        clean, canon_dict = gen_district_scenario()
        scenario_type = "district_context"
        
    # 5. FULL SERVICE (10%) - "Kyiv, Nova Poshta 5"
    elif scenario_roll < 0.95:
        clean, canon_dict = gen_service_scenario()
        scenario_type = "service_full"
        
    # 6. VILLAGE (5%)
    else:
        clean, canon_dict = gen_village_scenario()
        scenario_type = "village"

    canonical_str = format_target_string(canon_dict)
    return {"scenario": scenario_type, "clean_input": clean, "canonical": canonical_str}

if __name__ == "__main__":
    # Test run
    for i in range(5):
        sample = generate_single_sample()
        print(json.dumps(sample, ensure_ascii=False, indent=2))