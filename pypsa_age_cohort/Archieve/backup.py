import pandas as pd
import numpy as np
import os


#Read PyPSA dataset
df_PyPSA = pd.read_csv(
    "powerplants_18_02_2026.csv",
    sep=",",
    encoding="utf-8",
    low_memory=False,
    skipinitialspace=True,
    index_col=False,
)

print("✅ File loaded successfully!\n")
print(f"The PyPSA dataframe has {df_PyPSA.shape[0]} rows and {df_PyPSA.shape[1]} columns {df_PyPSA.shape}")

#read in defined RECC_technolgies
RECC_technologies = pd.read_excel("RECC_technologies.xlsx",sheet_name="RECC_technologies", index_col=False)
RECC_technologies_complete_params = pd.read_excel("RECC_technologies.xlsx",sheet_name="tech_complete_params", index_col=False)

#get unique combinations from Fueltype and Technology columns
unique_combinations = df_PyPSA[["Fueltype", "Technology"]].drop_duplicates().sort_values(by=["Fueltype", "Technology"], ascending=True).reset_index(drop=True)
print(f"Unique combinations of Fueltype and Technology: {len(unique_combinations)}\n")

#export to excel for manual mapping
with pd.ExcelWriter("unique_combinations.xlsx") as writer:
        unique_combinations.to_excel(writer, sheet_name="unique_combinations", index=False)
        RECC_technologies.to_excel(writer, sheet_name="RECC_technologies", index=False)
        RECC_technologies_complete_params.to_excel(writer, sheet_name="RECC_tech_complete_params", index=False)
print("✅ The dataset unique_combinations was exported to Excel to define technology mapping manually\n")
