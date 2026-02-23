import pandas as pd
import requests
import time
from difflib import SequenceMatcher

# === Load the Excel file ===
file_path = "/Users/marcelgeller/Work/RECC/RECC_Model/ODYM-master/pypsa_age_cohort/rows_without_datein.xlsx"
df_missing = pd.read_excel(file_path)
print(f"Loaded {len(df_missing)} rows without DateIn.")

# === Wikidata SPARQL endpoint ===
WIKIDATA_URL = "https://query.wikidata.org/sparql"

# Map Fueltype/Technology to Wikidata subclasses (for better filtering)
wikidata_type_mapping = {
    "Wind": ["Q11404"],        # wind power station
    "Solar": ["Q2479475"],     # solar power station
    "Hydro": ["Q11339"],       # hydroelectric power station
    "Nuclear": ["Q18853"],     # nuclear power station
    "Coal": ["Q160506"],       # coal-fired power station
    "Oil": ["Q160545"],        # oil-fired power station
    "Gas": ["Q109109"],        # gas-fired power station
    "Biomass": ["Q1988790"],   # biomass power station
    "Geothermal": ["Q268093"], # geothermal power station
    "Waste": ["Q2150366"],     # waste-to-energy plant
}

def similar(a, b):
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_commission_year(name, country, fueltype=None, technology=None):
    """
    Query Wikidata for the commissioning year of a power plant.
    Uses Name + Country + optional Fueltype/Technology filtering.
    Allows approximate name matching.
    """
    query = f"""
    SELECT ?plant ?plantLabel ?inception ?type WHERE {{
      ?plant wdt:P31/wdt:P279* ?type;
             rdfs:label ?plantLabel;
             wdt:P17 ?country_entity;
             wdt:P571 ?inception.
      ?country_entity rdfs:label "{country}"@en.
      FILTER(LANG(?plantLabel) = "en")
    }}
    """
    headers = {"Accept": "application/sparql-results+json"}
    
    try:
        response = requests.get(WIKIDATA_URL, params={"query": query}, headers=headers, timeout=15)
        data = response.json()
        best_match_year = None
        highest_score = 0
        
        for item in data["results"]["bindings"]:
            plant_label = item["plantLabel"]["value"]
            inception = item["inception"]["value"]
            type_id = item["type"]["value"].split("/")[-1]  # get Wikidata QID
            
            # Check fueltype/technology if provided
            if fueltype and fueltype in wikidata_type_mapping:
                if type_id not in wikidata_type_mapping[fueltype]:
                    continue
            
            # Compute name similarity
            score = similar(name, plant_label)
            if score > highest_score:
                highest_score = score
                best_match_year = int(inception[:4])
        
        # Require a minimum similarity to accept
        if highest_score >= 0.7:
            return best_match_year
        
    except Exception as e:
        print(f"Error querying {name}: {e}")
    
    return None

# === Apply the function to the dataframe ===
commission_years = []

for idx, row in df_missing.iterrows():
    name = row["Name"]
    country = row["Country"]
    fueltype = row.get("Fueltype")
    technology = row.get("Technology")
    
    year = get_commission_year(name, country, fueltype, technology)
    commission_years.append(year)
    print(f"{name} ({country}) -> {year}")
    time.sleep(0.5)  # polite delay

df_missing["DateIn"] = commission_years

# === Save results to a new Excel file ===
output_path = "/Users/marcelgeller/Work/RECC/RECC_Model/ODYM-master/pypsa_age_cohort/rows_without_datein_filled.xlsx"
df_missing.to_excel(output_path, index=False)
print(f"✅ Done! Dataset with estimated commissioning years saved to {output_path}")
