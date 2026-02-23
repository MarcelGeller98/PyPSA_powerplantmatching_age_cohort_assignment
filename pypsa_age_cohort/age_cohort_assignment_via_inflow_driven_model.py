import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


from odym import classes as msc
from odym import dynamic_stock_model as dsm

#from PyPSA_powerplant_dataprocessing import build_powerplant_dataset


# --- Load & prepare data ---
#df_v1 = build_powerplant_dataset()
historic_inflows = pd.read_excel("historic_inflows_19_02_2026.xlsx")
lifetimes = pd.read_excel("PyPSA_lifetimes_18_02_2026.xlsx")

# time and other lists
years = sorted(historic_inflows["DateIn"].astype(int).unique())    # length t=129 from 1898 to 2026 or alternatively: np.arange(1890,2027)
countries = list(historic_inflows["Country"].unique())             # length r=36 
technologies = list(historic_inflows["Technologies"].unique())     # length I=109

n_years = len(years)
n_countries = len(countries)
n_techs = len(technologies)
n_elements = 1  # dummy element 'Fe' (expand later if needed)

historic_inflows_rIc_df = (
    historic_inflows.set_index(["Country", "Technologies", "DateIn"])
    .reindex(
        pd.MultiIndex.from_product(
            [countries, technologies, years],
            names=["Country", "Technologies", "DateIn"]
        ),
        fill_value=0
    )
    .reset_index()
)

# Convert to numpy array of shape (r, I, c)
# Build ordered categorical to force correct sort order
historic_inflows_rIc = (
    historic_inflows_rIc_df
    .assign(
        Country=pd.Categorical(historic_inflows_rIc_df["Country"], categories=countries, ordered=True),
        Technologies=pd.Categorical(historic_inflows_rIc_df["Technologies"], categories=technologies, ordered=True),
        DateIn=pd.Categorical(historic_inflows_rIc_df["DateIn"], categories=years, ordered=True),
    )
    .sort_values(["Country", "Technologies", "DateIn"])
    .set_index(["Country", "Technologies", "DateIn"])["Capacity"]
    .unstack("DateIn")
    .values
    .reshape(n_countries, n_techs, n_years)
)

# Expand directly to (r, I, e, t, c) with values on diagonal t==c
historic_inflows_rIetc = np.zeros((n_countries, n_techs, n_elements, n_years, n_years))
for i_c in range(n_years):
    historic_inflows_rIetc[:, :, 0, i_c, i_c] = historic_inflows_rIc[:, :, i_c]

#reorder lifetimes such that they comply with technology order in list of technologies
lifetimes_ordered = np.zeros((len(technologies), 2))
for t, tech in enumerate(technologies):
    tech_row = lifetimes.loc[lifetimes["Technology"] == tech]
    lifetimes_ordered[t, 0] = tech_row["Lifetime"].values[0]
    lifetimes_ordered[t, 1] = tech_row["Standard deviation"].values[0]
mean_lifetimes = lifetimes_ordered[:,0]
std_lifetimes  = lifetimes_ordered[:,1]

print(f"years={n_years}, countries={n_countries}, technologies={n_techs}, elements={n_elements}")

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
    'IndexLetter': ['t', 'c', 'e', 'r', 'I']
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
# Inflow parameter: indices 'r,I,e,c' -> historic_inflows_rIetc shape (t, r,e, I)
Dyn_MFA_System.ParameterDict['Inflow'] = msc.Parameter(
    Name='Installed capacity', ID=1, P_Res=1,
    MetaData=None, Indices='r,I,e,t,c', Values=historic_inflows_rIetc, Unit='MW/yr'
)

# Lifetimes per technology
Dyn_MFA_System.ParameterDict['tau'] = msc.Parameter(
    Name='mean lifetime', ID=2, P_Res=1,
    MetaData=None, Indices='I', Values=mean_lifetimes, Unit='yr'
)
Dyn_MFA_System.ParameterDict['sigma'] = msc.Parameter(
    Name='std lifetime', ID=3, P_Res=1,
    MetaData=None, Indices='I', Values=std_lifetimes, Unit='yr'
)

# --- Flows & stocks definitions (include 'e' where desired) ---
# F_0_1 (manufacturing) uses inflows (t,r,I)
Dyn_MFA_System.FlowDict['F_0_1'] = msc.Flow(Name='Technology Manufacturing', P_Start=0, P_End=1, Indices='r,I,e,t,c', Values=None)

# F_1_0 (EoL) and S_1 include element axis 'e' -> indices 'r,I,e,t,c'
Dyn_MFA_System.FlowDict['F_1_0'] = msc.Flow(Name='EoL products', P_Start=1, P_End=0, Indices='r,I,e,t,c', Values=None)
Dyn_MFA_System.StockDict['S_1'] = msc.Stock(Name='Technology stock', P_Res=1, Type=0, Indices='r,I,e,t,c', Values=None)
Dyn_MFA_System.StockDict['dS_1'] = msc.Stock(Name='Technology stock change', P_Res=1, Type=1, Indices='r,I,e,t', Values=None)

# initialize arrays inside ODYM objects
Dyn_MFA_System.Initialize_FlowValues()
Dyn_MFA_System.Initialize_StockValues()

print("Consistency check:", Dyn_MFA_System.Consistency_Check())

# --- Assign inflow numeric array to F_0_1 (r,I,c,e) ---
Dyn_MFA_System.FlowDict['F_0_1'].Values = historic_inflows_rIetc

# --- Prepare preallocated arrays for DSM outputs (with element axis) ---
# O_C_array, Stock_array shapes must match 't,c,r,e,I' -> (r,I,e,t,c)
O_C_array  = np.zeros((n_countries, n_techs, n_elements, n_years, n_years))  # r,I,e,t,c
Stock_array = np.zeros((n_countries, n_techs, n_elements, n_years, n_years))  # r,I,e,t,c
DS_array   = np.zeros((n_countries, n_techs, n_elements, n_years))            # r,I,e,t # no cohort for stock change

print("Preallocated shapes:")
print("  O_C_array:", O_C_array.shape)
print("  Stock_array:", Stock_array.shape)
print("  DS_array:", DS_array.shape)

# --- Run DSM per (region, technology) and fill arrays ---
for i_r in range(len(countries)):
    for i_T in range(len(technologies)):

        inflow_series = historic_inflows_rIc[i_r, i_T, :]
        lt_mean = Dyn_MFA_System.ParameterDict['tau'].Values[i_T]
        lt_std  = Dyn_MFA_System.ParameterDict['sigma'].Values[i_T]

        DSM = dsm.DynamicStockModel(
            t=np.array(years),
            i=inflow_series,
            lt={'Type': 'Normal', 'Mean': [lt_mean], 'StdDev': [lt_std]}
        )

        S_c = DSM.compute_s_c_inflow_driven()
        O_C = DSM.compute_o_c_from_s_c()
        DS  = DSM.compute_stock_change()

        O_C_array[i_r, i_T, 0, :, :] = O_C
        Stock_array[i_r, i_T, 0, :, :] = S_c
        DS_array[i_r, i_T, 0, :] = DS


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

plot = True
if plot == True:
    ##############################################################################################################
    '''PLOTS'''
    ##############################################################################################################

    ############################################
    # 0. Indizes und Zeitachse
    ############################################

    # Select country index
    i_r = countries.index("Germany")

    # Time vector
    time = np.array(Dyn_MFA_System.Time_L)
    years = time

    ############################################
    # 1. Define broader technology groups
    ############################################

    tech_groups = [
        ("Solar|PV", ["Solar|PV"]),
        ("Solar|PV", ["Solar|unknown"]),
        ("Solar|PV", ["Solar|Reservoir|PP"]),
        ("Solar|PV", ["Solar|Steam Turbine|PP"]),
        #("Solar|CSP", ["Solar|CSP"]),
        ("Wind|Onshore", ["Wind|Onshore"]),
        ("Wind|Offshore", ["Wind|Offshore"]),
        ("Natural Gas", ["Natural Gas"]),
        ("Coal", ["Coal", "Lignite"]),
        ("Oil", ["Oil"]),
        ("Hydro", ["Hydro"]),
        #("Biomass", ["Biogas", "Biomass"]),
        #("Battery", ["Battery"]),
        #("Geothermal", ["Geothermal"]),
        #("Waste", ["Waste"]),
    ]


    def assign_tech_group(tech):
        for group, patterns in tech_groups:
            if any(pat in tech for pat in patterns):
                return group
        return "Other"


    ############################################
    # 2. Inflows for Germany
    ############################################

    aggregated_inflows = {}

    for i_T, tech in enumerate(technologies):
        group = assign_tech_group(tech)

        inflow_series = historic_inflows_rIetc[i_r, i_T, 0, :, :] / 1000
        inflow_series = inflow_series.sum(axis=-1)
        # Diagnostic: print Solar|PV related techs
        if "Solar|PV" in tech:
            print(f"Tech: {tech}, Group: {group}, Max inflow: {inflow_series.max():.4f}, Sum: {inflow_series.sum():.4f}")
        if group in aggregated_inflows:
            aggregated_inflows[group] += inflow_series
        else:
            aggregated_inflows[group] = inflow_series.copy()

    plt.figure(figsize=(12, 6))

    for group, series in aggregated_inflows.items():
        plt.plot(time, series, label=group)

    plt.title("Inflows by technology group for Germany")
    plt.xlabel("Year")
    plt.ylabel("Inflow [GW per yr]")
    plt.legend()
    plt.grid(True)
    plt.xlim(1950, 2026)
    plt.show()


    ############################################
    # 3. Stock for Germany
    ############################################

    aggregated_stock = {}

    for i_T, tech in enumerate(technologies):
        group = assign_tech_group(tech)

        stock_series = np.nansum(Stock_array[i_r, i_T, 0, :,:], axis=1) / 1000

        if group in aggregated_stock:
            aggregated_stock[group] += stock_series
        else:
            aggregated_stock[group] = stock_series.copy()

    plt.figure(figsize=(12, 6))

    for group, series in aggregated_stock.items():
        plt.plot(time, series, label=group)

    plt.xlabel("Year")
    plt.ylabel("Stock [GW]")
    plt.title("Total power plant stock in Germany by technology group")
    plt.legend()
    plt.grid(True)
    plt.xlim(1950, 2026)
    plt.show()


    ############################################
    # 4. Outflow for Germany
    ############################################

    aggregated_outflow = {}

    for i_T, tech in enumerate(technologies):
        group = assign_tech_group(tech)

        outflow_series = np.nansum(O_C_array[i_r, i_T, 0, :,:], axis=(1)) / 1000

        if group in aggregated_outflow:
            aggregated_outflow[group] += outflow_series
        else:
            aggregated_outflow[group] = outflow_series.copy()

    plt.figure(figsize=(12, 6))

    for group, series in aggregated_outflow.items():
        plt.plot(years, series, label=group)

    plt.title("Outflows by technology group for Germany")
    plt.xlabel("Year")
    plt.ylabel("Outflows [GW per yr]")
    plt.legend()
    plt.grid(True)
    plt.xlim(1950, 2026)
    plt.show()


    ########################################
    #2025 Stock Germany, aggregated
    ########################################

    # --- parameters (adapt if variable names differ) ---
    '''country_name = "Germany"
    year_target = 2025

    # --- find indices ---
    i_r = countries.index(country_name)
    # time might be floats; find closest index to target year
    i_t = np.argmin(np.abs(time - year_target))

    # lists of sub-types to aggregate
    get_hydro_technologies = [item for item in technologies if "Hydro" in item]
    hydro_subtypes = get_hydro_technologies
    
    get_gas_technologies = [item for item in technologies if "Natural Gas" in item]
    gas_subtypes = get_gas_technologies

    # --- build mapping from each technology to the group name
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
    print(df_out.to_string(index=False))'''




    ########################################
    #2025 stock Germany
    ########################################

    '''import numpy as np
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
    print(df_stock.to_string(index=False))'''

    ########################################
    #2025 age-cohort composition
    ########################################

    # --- Parameters ---
    target_year = 2015
    date = datetime.now().strftime("%d_%m_%Y")
    outpath_excel = f"AgeCohort_Stock_{target_year}_AllCountries_{date}.xlsx"
    outpath_csv = f"AgeCohort_Stock_{target_year}_AllCountries_{date}.csv"

    # --- Ensure years is numpy numeric array for robust matching ---
    years_arr = np.asarray(years, dtype=float)  # years came from historic_inflows.index

    # find index of target year (nearest)
    i_t = int(np.argmin(np.abs(years_arr - target_year))) #alternative: "i_t = years.index(target_year)"
    actual_year = int(years_arr[i_t])
    if actual_year != target_year:
        print(f"🆘 Warning: exact year {target_year} not found in `years`. Using closest year {actual_year} at index {i_t}.")

    # verify Stock_array shape
    n_regions, n_T, n_elements, n_t, n_cohort  = Stock_array.shape
    print("Stock_array shape:", Stock_array.shape)
    assert i_t < n_t, "target year index out of bounds for Stock_array"

    # slice for 2025: shape -> (regions,technologies, element, cohort)
    stock = Stock_array[:, :, :, i_t, :]  # (regions,technologies, element, cohort)

    # --- Build records: one row per (region, technology, cohort) with positive surviving stock ---
    records = []
    for i_r, region in enumerate(countries):
        for i_T, tech in enumerate(technologies):
            # iterate cohorts (installation years)
            for i_c, cohort_year in enumerate(years):
                # sum across element axis (handles multiple elements if present)
                stock_value = np.nansum(stock[i_r, i_T, :, i_c])
                # keep only operational capacity > 0
                if stock_value > 0:
                    records.append({
                        "Region": region,
                        "Technology": tech,
                        "Cohort_year": int(cohort_year),
                        f"Stock_in_{target_year}_MW": float(stock_value)
                    })

    # convert to DataFrame and sort for readability
    df_stock = pd.DataFrame.from_records(records)
    if df_stock.empty:
        print(f"No surviving stock found for {target_year} in any region/technology/cohort.")
    else:
        df_stock = df_stock.sort_values(["Region", "Technology", "Cohort_year"], ascending=[True, True, True])
        # compute the share of each cohort in the region-technology total (optional but useful)
        df_stock["Total_by_Reg_Tech"] = df_stock.groupby(["Region", "Technology"])[f"Stock_in_{target_year}_MW"].transform("sum")
        df_stock["Cohort_share_pct"] = (df_stock[f"Stock_in_{target_year}_MW"] / df_stock["Total_by_Reg_Tech"]) *100

        # drop the helper total column if you prefer only shares and values
        # df_stock = df_stock.drop(columns=["Total_by_Reg_Tech"])

        # preview
        pd.set_option("display.max_rows", 10)
        print("\nPreview of exported data (first rows):")
        print(df_stock.head(10).to_string(index=False))
        
        excel_export = False
        if excel_export == True:
            # export to Excel and CSV
            df_stock.to_excel(outpath_excel, index=False)
            df_stock.to_csv(outpath_csv, index=False)
            print(f"\n✅ Exported voth files with {len(df_stock)} rows to:\n  {os.path.abspath(outpath_excel)}\n  {os.path.abspath(outpath_csv)}")

    #plot wind_onshore age_cohort distribution of 2025 stock with published data from Fachagentur für Wind und Solar (https://www.wind-energie.de/fileadmin/redaktion/dokumente/publikationen-oeffentlich/themen/06-zahlen-und-fakten/20260115_Status_des_Windenergieausbaus_an_Land_Jahr_2025.pdf)
    actual_wind_onshore_stock_germany = pd.read_excel("wind_onshore_age_cohort_distribution_2025_FachagenturWindSolar.xlsx")
    actual_pct_share_cohort_onshore = actual_wind_onshore_stock_germany["Cohort_Share"].values *100

    df_model_pct_share_cohort_onshore_from_1999 = df_stock[(df_stock["Region"] == "Germany") & (df_stock["Technology"] == "Wind|Onshore|PP") & (df_stock["Cohort_year"] >= 1999)]
    df_model_pct_share_cohort_til_1999_onshore = df_stock[(df_stock["Region"] == "Germany") & (df_stock["Technology"] == "Wind|Onshore|PP") & (df_stock["Cohort_year"] <= 1999)]
    sum_1999 = df_model_pct_share_cohort_til_1999_onshore["Cohort_share_pct"].sum()
    df_model_pct_share_cohort_onshore_from_1999.loc[df_model_pct_share_cohort_onshore_from_1999["Cohort_year"] == 1999, "Cohort_share_pct"] = sum_1999    
    model_pct_share_cohort_onshore = df_model_pct_share_cohort_onshore_from_1999["Cohort_share_pct"].values

    plt.figure(figsize=(12, 6))
    plt.plot(range(1999, 1999 + len(model_pct_share_cohort_onshore)), model_pct_share_cohort_onshore, label="Model")
    plt.plot(range(1999, 1999 + len(actual_pct_share_cohort_onshore)), actual_pct_share_cohort_onshore, label="Actual")
    plt.title("Germany Wind Onshore Age Cohort Distribution of Inflow-Driven Model and Actual Data")
    plt.xlabel("Year")
    plt.ylabel("% of Age Cohort")
    plt.legend()
    plt.grid(True)
    plt.xlim(1999, target_year)
    plt.show()


    # create dataframe for total stock per technology and region for comparison with ISE_energy_charts and export as new excel sheet

    # aggregate all hydropowerplants
    # aggregate all gas power plants

    # calculate average lifetime per technology and region as well as for total Europe and export as new excel sheet
    # calculate and plot technology and country specific lifetime divergence from average EU lifetimes
    # compare 

    #map PyPSA to RECC technologies with exisiting param set
    PyPSA_RECC_mapping = pd.read_excel("PyPSA_RECC_mapping.xlsx")

    # Create a mapping dictionary from PyPSA_RECC_mapping
    mapping_dict = PyPSA_RECC_mapping.set_index("PySPA Technologies")["RECC Technologies"].to_dict()

    # Map the Technology column in df_stock
    df_stock["Technology"] = df_stock["Technology"].map(mapping_dict)
    # Check for unmapped technologies
    unmapped_technologies = df_stock[df_stock["Technology"].isna()]

    if not unmapped_technologies.empty:
        unmapped_list = unmapped_technologies["Technology"].index.tolist()
        raise ValueError(f"Mapping failed for the following technologies: {unmapped_list}")
    
    # Get the list of RECC technologies with complete parameter sets
    valid_technologies = PyPSA_RECC_mapping["RECC technologies with complete parameter sets"].dropna().unique()
    # Filter df_stock to keep only rows with valid technologies
    df_stock = df_stock[df_stock["Technology"].isin(valid_technologies)]
    #check
    if df_stock.empty:
        print("All rows were dropped from df_stock. No valid technologies remain.")
    else:
        print(f"Filtered df_stock contains {len(df_stock)} rows.")

    #sum same  technologies for same region and year 
    # --- Reaggregate after PyPSA → RECC technology mapping ---

    # 1. Drop the now-stale computed columns to avoid confusion
    df_stock = df_stock.drop(columns=["Total_by_Reg_Tech", "Cohort_share_pct"])

    # 2. Aggregate: sum stock for rows that now share the same (Region, Technology, Cohort_year)
    #    This handles cases where multiple PyPSA technologies mapped to the same RECC technology
    df_stock = (
        df_stock
        .groupby(["Region", "Technology", "Cohort_year"], as_index=False)[f"Stock_in_{target_year}_MW"]
        .sum()
    )

    # 3. Recompute Total_by_Reg_Tech and Cohort_share_pct on the clean aggregated data
    df_stock["Total_by_Reg_Tech"] = (
        df_stock.groupby(["Region", "Technology"])[f"Stock_in_{target_year}_MW"]
        .transform("sum")
    )
    df_stock["Cohort_share_pct"] = (
        df_stock[f"Stock_in_{target_year}_MW"] / df_stock["Total_by_Reg_Tech"]
    )

    # 4. Re-sort for readability
    df_stock = df_stock.sort_values(
        ["Region", "Technology", "Cohort_year"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    # 5. Sanity check: each (Region, Technology) should sum to ~100%
    check = df_stock.groupby(["Region", "Technology"])["Cohort_share_pct"].sum()
    assert np.allclose(check, 1, atol=1e-6), f"Cohort shares don't sum to 100% for some groups:\n{check[~np.isclose(check, 1)]}"
    print("✅ Cohort shares validated: all (Region, Technology) groups sum to 100%.")
    
    df_stock = df_stock.drop(columns=[f"Stock_in_{target_year}_MW", "Total_by_Reg_Tech"])
    print(f"Preview of exported data (first rows): {df_stock.head()}")

    ###################################################################
    #Cohort Distribution of selected region and Technology
    ###################################################################
    
    select_tech = "Solar|PV|Other (Not Elsewhere Specified)" #Wind|Onshore|Other (Not Elsewhere Specified), etc.

    '''RECC technology list - Choose one of thr following technologies:
    Biomass|Other (Not Elsewhere Specified)
    Biomass|w/ CCS
    Coal|Other (Not Elsewhere Specified)
    Coal|w/ CCS
    Gas|Combined Cycle|Other (Not Elsewhere Specified)
    Gas|Combined Cycle|w/ CCS
    Geothermal|Other (Not Elsewhere Specified)
    Hydro|Other (Not Elsewhere Specified)
    Nuclear|Other (Not Elsewhere Specified)
    Oil|Other (Not Elsewhere Specified)
    Solar|CSP
    Solar|PV|Other (Not Elsewhere Specified)
    Wind|Offshore|Other (Not Elsewhere Specified)
    Wind|Onshore|Other (Not Elsewhere Specified)
    '''

    select_regions = ["Germany", "Spain", "Austria","France"]  # list of countries
    select_year = 1990

    # Define the full range of years
    full_years = pd.Series(range(select_year, target_year + 1), name="Cohort_year")

    plt.figure(figsize=(12, 6))

    for region in select_regions:
        # Filter for this region
        filtered_data = df_stock[
            (df_stock["Region"] == region) &
            (df_stock["Technology"] == select_tech) &
            (df_stock["Cohort_year"] >= select_year)
        ].sort_values(by="Cohort_year", ascending=True)

        # Reindex to include all years, filling missing with 0
        pct_share_cohort_from_select_year = (
            filtered_data.set_index("Cohort_year")
            .reindex(full_years, fill_value=0)
            .reset_index()
        )

        pct_share_cohort = pct_share_cohort_from_select_year["Cohort_share_pct"].values

        plt.plot(full_years, pct_share_cohort, label=region)

    plt.title(f"{select_tech} Age Cohort Distribution using an Inflow-Driven Model")
    plt.xlabel("Year")
    plt.ylabel("Age Cohort Distribution [%]")
    plt.legend()
    plt.grid(True)
    plt.xlim(select_year, target_year)
    plt.show()


    excel_export_final = True
    if excel_export_final == True:
        #rename columns to fit RECC classifications
        df_stock.rename(columns= {"Region":"SSP_Regions_32","Technology":"Sectors_Industry", "Cohort_share_pct": f"Cohort_share_{target_year}"}, inplace=True)
        outpath_excel = f"3_SHA_RECC_Industry_AgeCohortDistribution_Stock{target_year}_{date}.xlsx"
        outpath_csv = f"3_SHA_RECC_Industry_AgeCohortDistribution_Stock{target_year}_{date}.csv"
        # export to Excel and CSV
        df_stock.to_excel(outpath_excel, index=False)
        #df_stock.to_csv(outpath_csv, index=False)
        print(f"\n✅ Exported voth files with {len(df_stock)} rows to:\n  {os.path.abspath(outpath_excel)}\n  {os.path.abspath(outpath_csv)}")