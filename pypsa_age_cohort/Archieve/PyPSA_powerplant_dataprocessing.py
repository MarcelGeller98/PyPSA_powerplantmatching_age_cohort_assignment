import pandas as pd
import numpy as np
import os


def build_powerplant_dataset():
    """
    Load, clean, and process the power plant dataset for RECC–ODYM integration.

    Returns
    -------
    df_v1 : pandas.DataFrame
        Processed dataset containing relevant columns and mapped technologies.
    """

    file_path = "/Users/marcelgeller/Work/RECC/Model Coupling/PyPSA_scenario_data_for_recc/powerplants 3.csv"

    # === Load dataset ===
    df = pd.read_csv(
        file_path,
        sep=",",
        encoding="utf-8",
        low_memory=False,
        skipinitialspace=True,
        index_col=False,
    )

    print("✅ File loaded successfully!")
    print(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns {df.shape}")

    # === Check for numeric values in selected columns ===
    columns_to_check = ["Capacity", "Efficiency", "DateIn", "DateRetrofit", "DateOut"]
    all_columns_ok = True
    for col in columns_to_check:
        total_rows = df.shape[0]
        num_floats = (df[col].apply(lambda x: isinstance(x, float))).sum()
        if num_floats != total_rows:
            print(f"⚠ {total_rows - num_floats} entries in '{col}' are NOT floats")
            all_columns_ok = False
    if all_columns_ok:
        print(f"\n✅ The columns {columns_to_check} contain only float values.")

    # === Check for missing categorical values ===
    columns_to_check = ["Name", "Fueltype", "Technology", "Set", "Country"]
    for col in columns_to_check:
        missing = df[col].isna().sum()
        empty_str = (df[col] == "").sum()
        total_empty = missing + empty_str
        if total_empty != 0:
            print(f"⚠ There are {total_empty} empty or missing entries in column '{col}'")

    # === Handle missing capacities ===
    mask_rows_without_capacity = (df["Capacity"] == 0) | (df["Capacity"] == "")
    rows_without_capacity = df[mask_rows_without_capacity]
    print(f"\nTotal entries without capacity values: {len(rows_without_capacity)}")

    latest_year_entry_row = rows_without_capacity.loc[
        rows_without_capacity["DateIn"].idxmax()
    ]
    latest_missing_year = rows_without_capacity["DateIn"].max()
    print(
        f"The latest plant without a capacity value is from year: {latest_missing_year}, "
        f"and is a {latest_year_entry_row['Fueltype']} plant"
    )

    define_year_to_drop = 1990
    if latest_missing_year < define_year_to_drop:
        df = df.drop(rows_without_capacity.index)
        print(
            f"All {len(rows_without_capacity)} rows without capacity values have been removed, "
            f"as the latest missing data is from before {define_year_to_drop}."
        )
    else:
        raise ValueError(
            f"🚨 Missing capacity data exists after {define_year_to_drop}. Please review manually."
        )

    individual_countries = rows_without_datein["Country"].unique()
    get_fueltypes = df["Fueltype"].unique()

    # === Handle missing DateIn values ===
    mask_rows_without_datein = (
        (df["DateIn"] == 0) | (df["DateIn"] == "") | (df["DateIn"].isna())
    )
    rows_without_datein = df[mask_rows_without_datein]
    print(f"\nTotal entries without DateIn values: {len(rows_without_datein)}")
    cumulative_capacity_of_missing_dateins = round(
        rows_without_datein["Capacity"].sum(), 1
    )
    print(
        f"The cumulative capacity of plants without given installation years is: "
        f"{cumulative_capacity_of_missing_dateins/1000} GW"
    )
    copy_rows_without_datein =rows_without_datein.sort_values(by='Country', ascending=True)


    copy_rows_without_datein.to_excel(
            "/Users/marcelgeller/Work/RECC/RECC_Model/ODYM-master/pypsa_age_cohort/rows_without_datein.xlsx",
            index=False
        )
    print("✅ rows_without_datein exported to Excel successfully!")

    # === Compute missing DateIn stats by country ===
    countries_missing_DateIn_capacity = pd.DataFrame(
        columns=["Country", "Missing_Capacity_[MW]"]
    )
    for country in individual_countries:
        sum_capacity = round(
            rows_without_datein.loc[rows_without_datein["Country"] == country, "Capacity"].sum(),
            1,
        )
        countries_missing_DateIn_capacity.loc[len(countries_missing_DateIn_capacity)] = [
            country,
            sum_capacity,
        ]

    countries_missing_DateIn_capacity = countries_missing_DateIn_capacity.sort_values(
        by="Missing_Capacity_[MW]", ascending=False, ignore_index=True
    )

    # Append fuel type breakdown
    for fuel in get_fueltypes:
        countries_missing_DateIn_capacity[fuel] = np.nan

    for country in individual_countries:
        for fuel in get_fueltypes:
            mask = (rows_without_datein["Country"] == country) & (
                rows_without_datein["Fueltype"] == fuel
            )
            sum_fueltype_capacity = round(
                rows_without_datein.loc[mask, "Capacity"].sum(), 1
            )
            countries_missing_DateIn_capacity.loc[
                countries_missing_DateIn_capacity["Country"] == country, fuel
            ] = sum_fueltype_capacity

    # Drop empty DateIn entries
    print(f"The current number of rows of df is {df.shape[0]}")
    df.drop(rows_without_datein.index, inplace=True)
    print(
        f"Now, after dropping empty DateIn entries, the current number of rows of df is {df.shape[0]}"
    )

    # === Technology mapping ===
    fuel_mapping = {
        "Biogas": "bio power plant",
        "Geothermal": "geothermal power plant",
        "Hard Coal": "hard coal power plant",
        "Lignite": "lignite coal power plant",
        "Nuclear": "nuclear power plant",
        "Oil": "oil power plant",
        "Other": "other",
        "Solid Biomass": "bio power plant",
        "Waste": "waste power plant",
    }

    tech_mapping = {
        # --- Solar ---
        ("Solar", "PV"): "solar photovoltaic power plant",
        ("Solar", "Csp"): "concentrating solar power plant",
        ("Solar", "Pv"): "solar photovoltaic power plant",
        ("Solar", "0"): "solar photovoltaic power plant",
        # --- Wind ---
        ("Wind", "Onshore"): "wind onshore power plant",
        ("Wind", "Offshore"): "wind offshore power plant",
        ("Wind", "0"): "wind onshore power plant",
        ("Wind", "Pv"): "wind onshore power plant",
        # --- Natural Gas ---
        ("Natural Gas", "Ccgt"): "combined cycle gas turbine power plant",
        ("Natural Gas", "Not Found"): "gas power plant, unspecified",
        ("Natural Gas", "Steam Turbine"): "gas steam turbine power plant",
        ("Natural Gas", "0"): "gas power plant, unspecified",
        ("Natural Gas", "Combustion Engine"): "gas power plant, unspecified",
        # --- Hydro ---
        ("Hydro", "Reservoir"): "hydro power plant, reservoir",
        ("Hydro", "Run-Of-River"): "hydro power plant, run-of-river",
        ("Hydro", "Pumped Storage"): "hydro power plant, pumped storage",
        ("Hydro", "Unknown"): "hydro power plant, unspecified",
        ("Hydro", "unknown"): "hydro power plant, unspecified",
        ("Hydro", "0"): "hydro power plant, unspecified",
    }

    # Clean whitespace and capitalization
    for col in ["Fueltype", "Technology", "Set"]:
        df[col] = df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)

    def map_technology(row):
        tech_key = (row["Fueltype"], str(row["Technology"]))
        if tech_key in tech_mapping:
            return tech_mapping[tech_key]
        elif row["Fueltype"] in fuel_mapping:
            return fuel_mapping[row["Fueltype"]]
        else:
            return "unknown technology"

    df["Technology"] = df["Technology"].fillna("0")
    df["Mapped_Technology"] = df.apply(map_technology, axis=1)

    # Drop irrelevant columns
    df.drop(
        columns=[
            "lat",
            "lon",
            "Duration",
            "Volume_Mm3",
            "DamHeight_m",
            "StorageCapacity_MWh",
            "EIC",
            "projectID",
        ],
        inplace=True,
    )

    count_unknown = df[df["Mapped_Technology"] == "unknown technology"]
    print(f"The number of unknown mapped technologies is: {count_unknown.shape[0]}")

    # Simplify dataset for inflow analysis
    df_v1 = df.drop(columns=["Name", "Technology", "Set", "Efficiency"])

    # Drop future inflows (after 2025)
    define_current_year = 2025
    rows_with_future_inflows = df_v1[df_v1["DateIn"] > define_current_year]
    df_v1.drop(rows_with_future_inflows.index, inplace=True)
    print(
        f"The dataset now (after removing future inflows) contains {df_v1.shape[0]} rows."
    )

    # Final cleaning and column ordering
    df_v1.drop(columns=["DateRetrofit", "DateOut"], inplace=True)
    df_v1.reset_index(drop=True, inplace=True)
    cols = list(df_v1.columns)
    cols.remove("Mapped_Technology")
    cols.insert(2, "Mapped_Technology")
    df_v1 = df_v1[cols]

    print(f"✅ Final dataset shape: {df_v1.shape}")
    return df_v1


# === Helper functions ===
def get_country_names(df_v1):
    """Return sorted list of unique countries."""
    return list(np.sort(df_v1["Country"].unique()))


def get_technology_names(df_v1):
    """Return sorted list of unique mapped technologies."""
    return list(np.sort(df_v1["Mapped_Technology"].unique()))


# === Allowing standalone execution ===
if __name__ == "__main__":
    df_v1 = build_powerplant_dataset()
    print(df_v1.head())
    print(f"Countries: {len(get_country_names(df_v1))}")
    print(f"Technologies: {len(get_technology_names(df_v1))}")
