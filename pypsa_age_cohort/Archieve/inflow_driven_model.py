import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from odym import classes as msc
from odym import dynamic_stock_model as dsm

from PyPSA_powerplant_dataprocessing import build_powerplant_dataset

# --- File paths (update if needed) ---
inflows_file_path = "/Users/marcelgeller/Work/RECC/RECC_Model/ODYM-master/pypsa_age_cohort/historic_inflows.xlsx"
lifetimes_file_path = "/Users/marcelgeller/Work/RECC/RECC_Model/ODYM-master/pypsa_age_cohort/lifetimes.xlsx"

# --- Load & prepare data ---
df_v1 = build_powerplant_dataset()
historic_inflows = pd.read_excel(inflows_file_path, header=[0, 1], index_col=0)
lifetimes = pd.read_excel(lifetimes_file_path, header=[0], index_col=0)

# time vector and index lists
years = historic_inflows.index.to_list()                 # length t
countries = list(historic_inflows.columns.levels[0])     # length r
technologies = list(historic_inflows.columns.levels[1])  # length T

n_years = len(years)
n_countries = len(countries)
n_techs = len(technologies)
n_elements = 1  # dummy element 'Fe' (expand later if needed)

# lifetimes (indexed by technology)
mean_lifetimes = lifetimes['average lifetime'].to_numpy()   # shape (T,)
std_lifetimes  = lifetimes['standard deviation'].to_numpy() # shape (T,)

print(f"years={n_years}, countries={n_countries}, technologies={n_techs}, elements={n_elements}")

# --- Convert MultiIndex inflows to 3D array (t, r, T) ---
# stacked: (t * r * T,) but reshape to (t, r, T)
stacked = historic_inflows.stack(level=[0,1])  # pandas may warn about future_stack; ignore
historic_inflows_3D = stacked.values.reshape(n_years, n_countries, n_techs)
print("historic_inflows_3D.shape:", historic_inflows_3D.shape)  # expect (t, r, T)
historic_inflows_4D = historic_inflows_3D[:, :, np.newaxis, :]  # shape -> (t,r,e,T)
print("historic_inflows_4D.shape:", historic_inflows_4D.shape)

# --- Build Model Classification & IndexTable (minimal, but consistent) ---
ModelClassification = {}
ModelClassification['Time'] = msc.Classification(Name='Time', Dimension='Time', ID=1, Items=years)
ModelClassification['Cohort'] = msc.Classification(Name='Age-cohort', Dimension='Time', ID=2, Items=years)
ModelClassification['Element'] = msc.Classification(Name='Elements', Dimension='Element', ID=3, Items=['Fe'])
ModelClassification['Region'] = msc.Classification(Name='Regions', Dimension='Region', ID=4, Items=countries)
ModelClassification['Technology'] = msc.Classification(Name='Technology', Dimension='Technology', ID=5, Items=technologies)

Model_Time_Start = int(min(years))
Model_Time_End = int(max(years))

IndexTable = pd.DataFrame({
    'Aspect': ['Time', 'Age-cohort', 'Element', 'Region', 'Technology'],
    'Description': ['time', 'age-cohort', 'element', 'region', 'technology'],
    'Dimension': ['Time', 'Time', 'Element', 'Region', 'Technology'],
    'Classification': [
        ModelClassification['Time'],
        ModelClassification['Cohort'],
        ModelClassification['Element'],
        ModelClassification['Region'],
        ModelClassification['Technology']
    ],
    'IndexLetter': ['t', 'c', 'e', 'r', 'T']
})
IndexTable.set_index('Aspect', inplace=True)

# --- Initialize MFA system ---
Dyn_MFA_System = msc.MFAsystem(
    Name='StockAccumulationSystem',
    Geogr_Scope='SelectedRegions',
    Unit='MW',
    ProcessList=[],
    FlowDict={},
    StockDict={},
    ParameterDict={},
    Time_Start=Model_Time_Start,
    Time_End=Model_Time_End,
    IndexTable=IndexTable,
    Elements=IndexTable.loc['Element'].Classification.Items
)

# --- Processes ---
Dyn_MFA_System.ProcessList = [msc.Process(Name='Manufacturing', ID=0),
                             msc.Process(Name='Use phase', ID=1)]
print("Processes:", [p.Name for p in Dyn_MFA_System.ProcessList])

# --- Parameters ---
# Inflow parameter: indices 't,r,e,T' -> historic_inflows_4D shape (t, r,e, T)
Dyn_MFA_System.ParameterDict['Inflow'] = msc.Parameter(
    Name='Installed capacity', ID=1, P_Res=1,
    MetaData=None, Indices='t,r,e,T', Values=historic_inflows_4D, Unit='MW/yr'
)

# Lifetimes per technology
Dyn_MFA_System.ParameterDict['tau'] = msc.Parameter(
    Name='mean lifetime', ID=2, P_Res=1,
    MetaData=None, Indices='T', Values=mean_lifetimes, Unit='yr'
)
Dyn_MFA_System.ParameterDict['sigma'] = msc.Parameter(
    Name='std lifetime', ID=3, P_Res=1,
    MetaData=None, Indices='T', Values=std_lifetimes, Unit='yr'
)

# --- Flows & stocks definitions (include 'e' where desired) ---
# F_0_1 (manufacturing) uses inflows (t,r,T)
Dyn_MFA_System.FlowDict['F_0_1'] = msc.Flow(Name='Technology Manufacturing', P_Start=0, P_End=1, Indices='t,r,e,T', Values=None)

# F_1_0 (EoL) and S_1 include element axis 'e' -> indices 't,c,r,e,T'
Dyn_MFA_System.FlowDict['F_1_0'] = msc.Flow(Name='EoL products', P_Start=1, P_End=0, Indices='t,c,r,e,T', Values=None)
Dyn_MFA_System.StockDict['S_1'] = msc.Stock(Name='Technology stock', P_Res=1, Type=0, Indices='t,c,r,e,T', Values=None)
Dyn_MFA_System.StockDict['dS_1'] = msc.Stock(Name='Technology stock change', P_Res=1, Type=1, Indices='t,r,e,T', Values=None)

# initialize arrays inside ODYM objects
Dyn_MFA_System.Initialize_FlowValues()
Dyn_MFA_System.Initialize_StockValues()

print("Consistency check:", Dyn_MFA_System.Consistency_Check())

# --- Assign inflow numeric array to F_0_1 (t,r,T) ---
Dyn_MFA_System.FlowDict['F_0_1'].Values = historic_inflows_4D

# --- Prepare preallocated arrays for DSM outputs (with element axis) ---
# O_C_array, Stock_array shapes must match 't,c,r,e,T' -> (t, c, r, e, T)
O_C_array = np.zeros((n_years, n_years, n_countries, n_elements, n_techs))
Stock_array = np.zeros((n_years, n_years, n_countries, n_elements, n_techs))
# dS shape must match 't,r,e,T'
DS_array = np.zeros((n_years, n_countries, n_elements, n_techs))

print("Preallocated shapes:")
print("  O_C_array:", O_C_array.shape)
print("  Stock_array:", Stock_array.shape)
print("  DS_array:", DS_array.shape)

# --- Run DSM per (region, technology) and fill arrays ---
for i_r, region in enumerate(countries):
    for i_T, tech in enumerate(technologies):
        # pick the singleton element index 0
        inflow_series = Dyn_MFA_System.ParameterDict['Inflow'].Values[:, i_r, 0, i_T]  # shape (t,)
        lt_mean = Dyn_MFA_System.ParameterDict['tau'].Values[i_T]
        lt_std  = Dyn_MFA_System.ParameterDict['sigma'].Values[i_T]

        DSM = dsm.DynamicStockModel(
            t=np.array(years),
            i=inflow_series,
            lt={'Type': 'Normal', 'Mean': [lt_mean], 'StdDev': [lt_std]}
        )

        S_c = DSM.compute_s_c_inflow_driven()   # shape (t, c)
        O_C = DSM.compute_o_c_from_s_c()        # shape (t, c)
        DS = DSM.compute_stock_change()         # shape (t,)

        O_C_array[:, :, i_r, 0, i_T] = O_C
        Stock_array[:, :, i_r, 0, i_T] = S_c
        DS_array[:, i_r, 0, i_T] = DS


# --- Assign modeled results into ODYM structures (shapes must match Indices) ---
Dyn_MFA_System.FlowDict['F_1_0'].Values = O_C_array
Dyn_MFA_System.StockDict['S_1'].Values   = Stock_array
Dyn_MFA_System.StockDict['dS_1'].Values  = DS_array

# --- Sanity print: show each ODYM object's declared Indices and actual array shape ---
def print_obj_shape(name, obj):
    vals = obj.Values
    shape = None if vals is None else getattr(vals, "shape", None)
    print(f"{name}: Indices='{obj.Indices}', assigned shape={shape}")

print("\nAssigned objects (Indices vs assigned NumPy shapes):")
print_obj_shape("F_0_1", Dyn_MFA_System.FlowDict['F_0_1'])
print_obj_shape("F_1_0", Dyn_MFA_System.FlowDict['F_1_0'])
print_obj_shape("S_1",   Dyn_MFA_System.StockDict['S_1'])
print_obj_shape("dS_1",  Dyn_MFA_System.StockDict['dS_1'])


print("\n=== ODYM Flow & Stock Diagnostic ===\n")

# Check all flows
for key, flow in Dyn_MFA_System.FlowDict.items():
    values = flow.Values
    if values is None:
        print(f"Flow '{key}' has no Values assigned!")
    else:
        n_nan = np.isnan(values).sum()
        print(f"Flow '{key}': shape={values.shape}, n_nan={n_nan}")

# Check all stocks
for key, stock in Dyn_MFA_System.StockDict.items():
    values = stock.Values
    if values is None:
        print(f"Stock '{key}' has no Values assigned!")
    else:
        n_nan = np.isnan(values).sum()
        print(f"Stock '{key}': shape={values.shape}, n_nan={n_nan}")

# Check DSM arrays if you preallocated separately
arrays_to_check = {
    "O_C_array": O_C_array,
    "Stock_array": Stock_array,
    "DS_array": DS_array
}
for name, arr in arrays_to_check.items():
    n_nan = np.isnan(arr).sum()
    print(f"{name}: shape={arr.shape}, n_nan={n_nan}")


# --- Run MassBalance (should not raise the 'e' einsum error) ---
Bal = Dyn_MFA_System.MassBalance()
print("\nMass balance results:")
print("Balance shape:", Bal.shape)
print("Sum of absolute balancing errors by process:", np.abs(Bal).sum(axis=0))

print("\n✅ Script finished successfully.")


##############################################################################################################
'''PLOTS'''
##############################################################################################################


# Technologies of interest
techs_of_interest = [
    'solar photovoltaic power plant',
    'wind onshore power plant',
    'wind offshore power plant',
    'lignite coal power plant',
    'nuclear power plant'
]

# Select Country index
i_r = countries.index('Germany')

# Get technology indices
tech_indices = [technologies.index(t) for t in techs_of_interest]

# Time vector
time = np.array(Dyn_MFA_System.Time_L)


##########################
'''Inflows for Germany'''
##########################

plt.figure(figsize=(12,6))

for i_T, tech in zip(tech_indices, techs_of_interest):
    inflow_series = historic_inflows_4D[:, i_r, 0, i_T]/1000  # shape (t,)
    plt.plot(time, inflow_series, label=tech)

plt.title("Inflows by technology for Germany")
plt.xlabel("Year")
plt.ylabel("Inflow [GW/yr]")
plt.legend()
plt.grid(True)
plt.xlim(1950, 2026) 
plt.show()

##########################
'''Stock for Germany'''
##########################
plt.figure(figsize=(12, 6))
for i_T, tech in zip(tech_indices, techs_of_interest):
    # Sum across age-cohort axis (1) and element axis (3)
    stock_series = np.nansum(Stock_array[:, :, i_r, 0, i_T], axis=1)/1000
    plt.plot(time, stock_series, label=tech)

plt.xlabel('Year')
plt.ylabel('Stock [GW]')
plt.title('Total power plant stock in Germany by technology')
plt.legend()
plt.grid(True)
plt.xlim(1950, 2026)
plt.show()


##########################
'''Outflow for Germany'''
##########################
for i_T, tech in zip(tech_indices, techs_of_interest):
    # Sum across cohort (axis=1) and element (axis=2)
    outflow_series = np.nansum(O_C_array[:, :, i_r, :, i_T], axis=(1, 2))/1000
    plt.plot(years, outflow_series, label=tech)

plt.title("Outflows by Technology for Germany")
plt.xlabel("Year")
plt.ylabel("Outflows [GW/year]")
plt.legend()
plt.grid(True)
plt.xlim(1950, 2026)
plt.show()

########################################
'''2025 Stock Germany, aggregated'''
########################################

# --- parameters (adapt if variable names differ) ---
country_name = "Germany"
year_target = 2025

# --- find indices ---
i_r = countries.index(country_name)
# time might be floats; find closest index to target year
i_t = np.argmin(np.abs(time - year_target))

# lists of sub-types to aggregate
hydro_subtypes = [
    "hydro power plant, reservoir",
    "hydro power plant, run-of-river",
    "hydro power plant, pumped storage",
    "hydro power plant, unspecified"
]
gas_subtypes = [
    "combined cycle gas turbine power plant",
    "gas power plant, unspecified",
    "gas steam turbine power plant"
]

# --- build mapping from each technology to the group name (or itself) ---
tech_to_group = {}
for tech in technologies:
    if tech in hydro_subtypes:
        tech_to_group[tech] = "Hydro power plant"
    elif tech in gas_subtypes:
        tech_to_group[tech] = "Natural gas power plant"
    else:
        tech_to_group[tech] = tech  # keep original name

# --- invert mapping: group -> list of tech indices ---
group_to_indices = {}
for i_T, tech in enumerate(technologies):
    group = tech_to_group[tech]
    group_to_indices.setdefault(group, []).append(i_T)

# --- compute total stock in year_target per group ---
group_totals = {}
for group, idx_list in group_to_indices.items():
    total = 0.0
    for i_T in idx_list:
        # sum over cohorts (axis=1) and elements (axis=3) for that technology
        # Stock_array shape: (t_current, t_cohort, region, element, technology)
        # Extract Stock_array[i_t, :, i_r, :, i_T] -> shape (cohort, element)
        stock_by_cohort_and_element = Stock_array[i_t, :, i_r, :, i_T]
        # sum across cohorts and elements; treat NaNs as zero
        total += np.nansum(stock_by_cohort_and_element)
    group_totals[group] = total

# --- prepare for plotting: keep order predictable (sorted by value descending) ---
groups = list(group_totals.keys())
values = np.array([group_totals[g] for g in groups])

# sort by value descending for nicer plot
order = np.argsort(values)[::-1]
groups_sorted = [groups[i] for i in order]
values_sorted = values[order]/1000

# --- bar plot ---
plt.figure(figsize=(12,6))
bars = plt.bar(groups_sorted, values_sorted)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Installed stock in {} [GW]".format(year_target))
plt.title(f"Installed stock in {country_name} by technology group — {year_target}")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- print table for inspection and export to Excel if wanted ---
df_out = pd.DataFrame({
    "Technology Group": groups_sorted,
    "Stock_MW": values_sorted
})
print(df_out.to_string(index=False))


########################################
'''2025 stock Germany'''
########################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- parameters ---
country_name = "Germany"
year_target = 2025

# --- find indices ---
i_r = countries.index(country_name)
i_t = np.argmin(np.abs(time - year_target))  # nearest index to 2025

# --- compute total stock per technology (summing over cohorts & elements) ---
tech_totals = []
for i_T, tech in enumerate(technologies):
    stock_slice = Stock_array[i_t, :, i_r, :, i_T]  # shape (cohort, element)
    total_stock = np.nansum(stock_slice)/1000  # total stock in that year for that tech
    tech_totals.append(total_stock)

# --- convert to DataFrame for convenience ---
df_stock = pd.DataFrame({
    "Technology": technologies,
    "Stock_MW": tech_totals
})

# --- sort descending for clarity ---
df_stock = df_stock.sort_values(by="Stock_MW", ascending=False)

# --- plot ---
plt.figure(figsize=(12,6))
plt.bar(df_stock["Technology"], df_stock["Stock_MW"], color="steelblue")
plt.xticks(rotation=60, ha="right")
plt.ylabel("Installed Stock [GW]")
plt.title(f"Installed Stock by Technology in {country_name} ({year_target})")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- optional printout for inspection ---
print(df_stock.to_string(index=False))

########################################
'''2025 age-cohort composition'''
########################################

# --- Parameters ---
target_year = 2024
outpath_excel = "AgeCohort_Stock_2024_AllCountries.xlsx"  # change as desired
outpath_csv = "AgeCohort_Stock_2024_AllCountries.csv"

# --- Ensure years is numpy numeric array for robust matching ---
years_arr = np.asarray(years, dtype=float)  # years came from historic_inflows.index

# find index of target year (nearest)
i_t = int(np.argmin(np.abs(years_arr - target_year))) #alternative: "i_t = years.index(target_year)"
actual_year = int(years_arr[i_t])
if actual_year != target_year:
    print(f"Warning: exact year {target_year} not found in `years`. Using closest year {actual_year} at index {i_t}.")

# verify Stock_array shape
n_t, n_cohort, n_regions, n_elements, n_T = Stock_array.shape
print("Stock_array shape:", Stock_array.shape)
assert i_t < n_t, "target year index out of bounds for Stock_array"

# slice for 2025: shape -> (cohort, regions, elements, technologies)
stock_2025 = Stock_array[i_t, :, :, :, :]  # (cohort, region, element, technology)

# --- Build records: one row per (region, technology, cohort) with positive surviving stock ---
records = []
for i_r, region in enumerate(countries):
    for i_T, tech in enumerate(technologies):
        # iterate cohorts (installation years)
        for i_c, cohort_year in enumerate(years):
            # sum across element axis (handles multiple elements if present)
            stock_value = np.nansum(stock_2025[i_c, i_r, :, i_T])
            # keep only operational capacity > 0
            if stock_value > 0:
                records.append({
                    "Region": region,
                    "Technology": tech,
                    "Cohort_year": int(cohort_year),
                    "Stock_in_2024_MW": float(stock_value)
                })

# convert to DataFrame and sort for readability
df_stock_2025 = pd.DataFrame.from_records(records)
if df_stock_2025.empty:
    print("No surviving stock found for 2025 in any region/technology/cohort.")
else:
    df_stock_2025 = df_stock_2025.sort_values(["Region", "Technology", "Cohort_year"], ascending=[True, True, True])
    # compute the share of each cohort in the region-technology total (optional but useful)
    df_stock_2025["Total_by_Reg_Tech"] = df_stock_2025.groupby(["Region", "Technology"])["Stock_in_2024_MW"].transform("sum")
    df_stock_2025["Cohort_share_pct"] = (df_stock_2025["Stock_in_2024_MW"] / df_stock_2025["Total_by_Reg_Tech"]) * 100.0

    # drop the helper total column if you prefer only shares and values
    # df_stock_2025 = df_stock_2025.drop(columns=["Total_by_Reg_Tech"])

    # preview
    pd.set_option("display.max_rows", 10)
    print("\nPreview of exported data (first rows):")
    print(df_stock_2025.head(10).to_string(index=False))

    # export to Excel and CSV
    df_stock_2025.to_excel(outpath_excel, index=False)
    df_stock_2025.to_csv(outpath_csv, index=False)
    print(f"\n✅ Exported {len(df_stock_2025)} rows to:\n  {os.path.abspath(outpath_excel)}\n  {os.path.abspath(outpath_csv)}")

# create dataframe for total stock per technology and region for comparison with ISE_energy_charts and export as new excel sheet

# aggregate all hydropowerplants
# aggregate all gas power plants

# calculate average lifetime per technology and region as well as for total Europe and export as new excel sheet
# calculate and plot technology and country specific lifetime divergence from average EU lifetimes 
# compare 
