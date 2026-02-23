import pandas as pd
import numpy as np
import os
from datetime import datetime

################################################################################################################################################################################################################################################################################################################################################################################
#   Technology Naming Convention
#   The technology names follow a hierarchical naming convention, where the | separator indicates increasing
#       levels of specificity. Each level to the right of a | provides a more detailed description of the technology
#       named to its left.
#   General Structure:
#   Technology Category | Sub-type | Configuration | Variant
#
#   Rules:
#   1. Hierarchy depth varies by technology Not all technologies are subdivided to the same depth.
#       A technology is only broken down further if meaningful distinctions exist:
#       •	Geothermal|Other (Not Elsewhere Specified) — stops at Level 2
#       •	Wind|Onshore|PMSG-DD — goes to Level 3
#       •	Solar|PV|Rooftop|SHJ — goes to Level 4
#
#   2. "Other (Not Elsewhere Specified)" as a catch-all When a category is subdivided into explicitly named variants,
#   an additional "Other (Not Elsewhere Specified)" entry is added at the same level to capture any technology that does
#   not fit the named subcategories. If a category is not subdivided, no "Other" entry is added.
#
#   3. Entries can exist at any level A technology can appear at any level of the hierarchy without
#   requiring further subdivision, as long as it represents a sufficiently distinct and self-contained
#   category (e.g., Solar|CSP, Geothermal|Other (Not Elsewhere Specified)).
################################################################################################################################################################################################################################################################################################################################################################################



def build_powerplant_dataset():
    """
    Load, clean, and process the power plant dataset for RECC–ODYM integration.

    Returns
    -------
    Processed dataset (pandas.DataFrame) containing relevant columns and mapped technologies.
    """

    file_path = "/Users/marcelgeller/Work/RECC/RECC_Model/ODYM-master/pypsa_age_cohort/powerplants_18_02_2026.csv"

    #Load dataset
    df = pd.read_csv(
        file_path,
        sep=",",
        encoding="utf-8",
        low_memory=False,
        skipinitialspace=True,
        index_col=False,
    )

    print("✅ File loaded successfully!\n")
    print(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns {df.shape}")

    #drop unnecessary columns
    df = df.iloc[:,1:11].drop(columns=["Efficiency"]) #drops id column and all columns after DateOut + Efficiency column

    #Check for numeric values in selected columns
    columns_to_check = ["Capacity", "DateIn", "DateRetrofit", "DateOut"]
    all_columns_ok = True
    for col in columns_to_check:
        total_rows = df.shape[0]
        num_floats = (df[col].apply(lambda x: isinstance(x, float))).sum()
        if num_floats != total_rows:
            print(f"⚠ {total_rows - num_floats} entries in '{col}' are NOT floats")
            all_columns_ok = False
    if all_columns_ok:
        print(f"\n✅ The columns {columns_to_check} contain only float values.\n")

    #Check for missing categorical values
    columns_to_check = ["Name", "Fueltype", "Technology", "Set", "Country"]
    for col in columns_to_check:
        missing = df[col].isna().sum()
        empty_str = (df[col] == "").sum()
        total_empty = missing + empty_str
        if total_empty != 0:
            print(f"⚠ There are {total_empty} empty or missing entries in column '{col}'")

    #Drop entries with missing capacities
    mask_rows_without_capacity = (df["Capacity"] == 0) | (df["Capacity"] == "") | (df["Capacity"].isna())
    rows_without_capacity = df[mask_rows_without_capacity]
    print(f"\nTotal entries without capacity values: {len(rows_without_capacity)}")
    df = df.drop(rows_without_capacity.index)
    print(f"All {len(rows_without_capacity)} rows without capacity values have been removed\n")

    #Handle missing DateIn values
    mask_rows_without_datein = ((df["DateIn"] == 0) | (df["DateIn"] == "") | (df["DateIn"].isna()))
    rows_without_datein = df[mask_rows_without_datein]
    print(f"Total entries without DateIn values: {len(rows_without_datein)}")
    cumulative_capacity_of_missing_dateins = round(rows_without_datein["Capacity"].sum(), 1)
    print(f"The cumulative capacity of plants without given installation years is: {round(cumulative_capacity_of_missing_dateins/1000,1)}GW\n")
    copy_rows_without_datein =rows_without_datein.sort_values(by='Country', ascending=True)

    #Compute missing DateIn stats by country
    individual_countries_without_datein = rows_without_datein["Country"].unique()
    countries_missing_DateIn_capacity = pd.DataFrame(columns=["Country", "Missing_Capacity_[MW]"])
    for country in individual_countries_without_datein:
        sum_capacity = round(rows_without_datein.loc[rows_without_datein["Country"] == country, "Capacity"].sum(), 1,)
        countries_missing_DateIn_capacity.loc[len(countries_missing_DateIn_capacity)] = [country, sum_capacity,]

    countries_missing_DateIn_capacity = countries_missing_DateIn_capacity.sort_values(by="Missing_Capacity_[MW]", ascending=False, ignore_index=True)

    # Append fuel type breakdown
    get_fueltypes = df["Fueltype"].unique()
    for fuel in get_fueltypes:
        countries_missing_DateIn_capacity[fuel] = np.nan

    for country in individual_countries_without_datein:
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

    export = False 
    if export == True:
        with pd.ExcelWriter("rows_without_datein.xlsx") as writer:
            copy_rows_without_datein.to_excel(writer, sheet_name="rows_without_datein", index=False)
            countries_missing_DateIn_capacity.to_excel(writer, sheet_name="stats_by_country", index=False)
    #print("✅ The dataframe rows_without_datein and countries_missing_DateIn_capacity have successfully been exported to Excel for inspection if desired.\n")

    #Drop future inflows
    current_year = datetime.now().year

    #exlude all entries with DateIn values > current_year
    mask_future_inflows = df["DateIn"] > current_year
    rows_with_future_inflows = df[mask_future_inflows]
    print(f"The number of entries with future DateIn values (after {current_year}) is: {len(rows_with_future_inflows)}")
    print(f"The current number of rows of df is {df.shape[0]}")
    df.drop(rows_with_future_inflows.index, inplace=True)
    print(f"After dropping future inflows, the dataset now contains {df.shape[0]} rows\n")
    df.drop(rows_without_datein.index, inplace=True)
    print(f"Now, after dropping empty DateIn entries, the dataset reduces to {df.shape[0]} rows\n")

    #combine strings from column Fueltype, Technology and Set to insert into a new column "Technologies complete information"
    df.insert(
        loc=4,
        column="Technologies",
        value=(
            df["Fueltype"].fillna("unknown fuel type").astype(str)
            + "|"
            + df["Technology"].fillna("unknown technology").astype(str)
            + "|"
            + df["Set"].fillna("unknown set").astype(str)
        )
    )

    #remove capacities which have been decommissioned already (DateOut < current_year)
    mask_decommissioned = df["DateOut"] < current_year
    rows_decommissioned = df[mask_decommissioned]
    print(f"The number of entries with DateOut values before {current_year} is: {len(rows_decommissioned)}")
    print(f"The current number of rows of df is {df.shape[0]}")
    df.drop(rows_decommissioned.index, inplace=True)
    print(f"After dropping decommissioned plants, the dataset now contains {df.shape[0]} rows\n")

    df = df.iloc[:,4:]
    df.drop(columns=["DateRetrofit", "DateOut"], inplace=True)

    # Final cleaning and column ordering
    df.reset_index(drop=True, inplace=True)
    cols = list(df.columns)
    cols.remove("Technologies")
    cols.insert(1, "Technologies")
    cols.remove("DateIn")
    cols.insert(2, "DateIn")
    df = df[cols].sort_values(by=["Country", "Technologies", "DateIn"], ascending=True).reset_index(drop=True)

    #aggregate df by country, technology and DateIn
    df = df.groupby(["Country", "Technologies", "DateIn"], as_index=False).sum()

    print(f"✅ Final dataset shape: {df.shape}\n")
    return df

export_to_excel = True
if export_to_excel == True:
    df = build_powerplant_dataset()
    date = datetime.now().strftime("%d_%m_%Y")
    df.to_excel(f"historic_inflows_{date}.xlsx", index=False)
    print("✅ The processed dataset has been exported to Excel successfully!")

# === Helper functions ===
def get_country_names(df):
    """Return sorted list of unique countries."""
    return list(np.sort(df["Country"].unique()))


def get_technology_names(df):
    """Return sorted list of unique mapped technologies."""
    return list(np.sort(df["Technologies"].unique()))


# === Allowing standalone execution ===
if __name__ == "__main__":
    df = build_powerplant_dataset()
    print(f"Inspection of first dataframe entries\n{df.head()}")
    print(f"\nCountries: {len(get_country_names(df))}")
    print(f"Technologies: {len(get_technology_names(df))}")
