import pandas as pd
import requests
from io import StringIO
import re
import numpy as np
import os
import tabula

from classification.filters import normalize_cas


"""
Module for preparing IPCC data of GHGs for matching
"""
def prepare_ar6_data():
    """Fetch and enrich AR6 GHG data, returning the enriched dataframe."""
    # Step 1: Fetch IPCC AR6 GHG data from GitHub
    df_ar6, ar6_cols = fetch_ar6_ghg_data()

    # Step 2: Add manual Radiative efficiency entries for CO2, CH4, N2O
    df_ar6 = enrich_ar6_with_manual_ghg(df_ar6, ar6_cols)

    # Step 3: Fetch and enrich with Hodnebrog et al. (2020) data
    df_hodne, hodne_cols = fetch_hodnebrog_data()

    # Enrich AR6 data with Radiative efficiency from Hodnebrog
    df_ar6, _ = enrich_ar6_with_hodnebrog(df_ar6, df_hodne, ar6_cols, hodne_cols)

    return df_ar6, ar6_cols


def prepare_wmo_data(pdf_path: str = None):
    """
    Extract and prepare WMO 2022 Ozone Assessment data from Table A-5.
    
    Args:
        pdf_path: Path to the WMO PDF file. If None, uses default path.
        
    Returns:
        Tuple of (dataframe, column_names_dict)
        - dataframe: DataFrame with WMO data
        - column_names: Dict with keys: 'name', 'formula', 'cas', 'lifetime', 'odp', 'rad_eff'
    """
    
    # Default PDF path if not provided
    if pdf_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(os.path.dirname(script_dir), "data", "01_Scientific assessment of ozone depletion 2022.pdf")
    
    # Helper functions
    def to_float(x):
        """Convert a string to float"""
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace("–", "-").replace("−", "-")
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else np.nan
    
    def parse_lifetime_years(x):
        """Convert a lifetime string to years (handle days/months, assume years otherwise)"""
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if not s:
            return np.nan
        s = s.replace("–", "-").replace("−", "-")
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if not m:
            return np.nan
        val = float(m.group(1))
        s_low = s.lower()
        if "day" in s_low:
            return val / 365.25
        if "month" in s_low:
            return val / 12.0
        return val  # assume years
    
    def make_unique_columns(df):
        """Make column names unique by appending .1, .2, etc. to duplicates"""
        cols = df.columns.astype(str)
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df = df.copy()
        df.columns = new_cols
        return df
    
    # Table A-5 spans from page 458-493
    PAGES = "458-493"
    
    # Parse the PDF table into a list of dataframes
    dfs = tabula.read_pdf(
        pdf_path,
        pages=PAGES,
        multiple_tables=True,
        lattice=True,
        guess=True,
        java_options=["-Djava.awt.headless=true"],
    )
    
    # The tables are spanned over two pages, so we pair them up
    paired = [
        pd.concat([dfs[i].reset_index(drop=True), dfs[i+1].reset_index(drop=True)], axis=1)
        for i in range(0, len(dfs), 2)
    ]
    
    # The dataframes are prepared for concatenation by making their column names unique
    paired_fixed = [make_unique_columns(df) for df in paired]
    
    # Concatenate all the paired tables into one long dataframe
    # Skip the first row of each subsequent table to avoid repeating header rows
    df = pd.concat(
        [paired_fixed[0]] + [df.iloc[1:] for df in paired_fixed[1:]],
        axis=0,
        ignore_index=True
    )
    
    # Turn pure-whitespace strings into NaN in all columns
    tmp = df.replace(r"^\s*$", np.nan, regex=True)
    
    # Remove the headline rows
    df = df.loc[~tmp.isna().all(axis=1)].reset_index(drop=True)
    
    # Only keep the columns needed and rename them
    keep_idx = [0, 1, 2, 5, 8, 9]
    df = df.iloc[:, keep_idx].copy()
    df.columns = [
        "Name",
        "Formula",
        "CAS",
        "WMO (2022) Total lifetime (years)",
        "ODP",
        "Radiative Efficiency (well mixed) (W m–2 ppb–1)",
    ]
    
    # Parse + clean the final columns
    
    # 1) Clean CAS: keep only valid CAS format, else NaN
    df["CAS"] = df["CAS"].astype(str).str.strip()
    df.loc[~df["CAS"].str.match(r"^\d{2,7}-\d{2}-\d$"), "CAS"] = np.nan
    
    # 2) Lifetime -> years (convert days/months to years, etc.)
    df["WMO (2022) Total lifetime (years)"] = df["WMO (2022) Total lifetime (years)"].apply(parse_lifetime_years)
    
    # 3) ODP -> float
    df["ODP"] = df["ODP"].apply(to_float)
    
    # 4) Radiative efficiency (well mixed) -> float
    df["Radiative Efficiency (well mixed) (W m–2 ppb–1)"] = (
        df["Radiative Efficiency (well mixed) (W m–2 ppb–1)"].apply(to_float)
    )
    
    # 5) Clean text columns
    df["Name"] = df["Name"].astype(str).str.strip().replace({"": np.nan})
    df["Formula"] = df["Formula"].astype(str).str.strip().replace({"": np.nan})
    
    # 6) Reset continuous index
    df.reset_index(drop=True, inplace=True)
    
    # 7) Merge rows with the same CAS: keep name/formula of first, max of other columns
    has_cas = df["CAS"].notna()
    with_cas = df[has_cas].copy()
    no_cas = df[~has_cas].copy()
    if not with_cas.empty:
        merged = (
            with_cas.groupby("CAS", as_index=False)
            .agg({
                "Name": "first",
                "Formula": "first",
                "WMO (2022) Total lifetime (years)": "max",
                "ODP": "max",
                "Radiative Efficiency (well mixed) (W m–2 ppb–1)": "max",
            })
        )
        df = pd.concat([merged, no_cas], ignore_index=True)
        df = df.reset_index(drop=True)
    
    # Create column names dictionary matching the AR6 pattern
    column_names = {
        'name': 'Name',
        'formula': 'Formula',
        'cas': 'CAS',
        'lifetime': 'WMO (2022) Total lifetime (years)',
        'odp': 'ODP',
        'rad_eff': 'Radiative Efficiency (well mixed) (W m–2 ppb–1)'
    }
    
    return df, column_names


"""
Module for fetching external data sources (AR6, Hodnebrog)
"""

def detect_column(df: pd.DataFrame, keywords: list) -> str:
    """
    Find column name in dataframe by matching against keywords (case-insensitive)
    
    Args:
        df: DataFrame to search
        keywords: List of keywords to match
        
    Returns:
        Column name if found, None otherwise
    """
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword.lower() in col_lower for keyword in keywords):
            return col
    return None


def fetch_ar6_ghg_data(keep_columns: int = 6) -> tuple:
    """
    Fetch IPCC AR6 GHG properties from GitHub
    
    Returns:
        Tuple of (dataframe, column_names_dict)
        - dataframe: DataFrame with AR6 GHG data
        - column_names: Dict with keys: 'name', 'cas', 'lifetime', 'rad_eff', 'formula'
    """
    
    csv_url = "https://raw.githubusercontent.com/chrisroadmap/ar6/main/data_input/ghg_properties/metrics_supplement.csv"
    response = requests.get(csv_url)
    response.raise_for_status()
    
    df_ar6 = pd.read_csv(StringIO(response.text))
    
    # Keep only first N columns
    df_ar6 = df_ar6.iloc[:, :keep_columns]
    
    
    # Detect column names (case-insensitive)
    name_col = detect_column(df_ar6, ['name'])
    cas_col = detect_column(df_ar6, ['cas'])
    lifetime_col = detect_column(df_ar6, ['lifetime'])
    rad_eff_col = detect_column(df_ar6, ['radiative', 'rad'])
    formula_col = detect_column(df_ar6, ['formula'])
    
    if cas_col is None:
        print("\n ERROR: Could not find CAS column. Available columns:")
        for col in df_ar6.columns:
            print(f"  - {col}")
        raise ValueError("CAS column not found in CSV")
    
    # Rename columns for clarity
    rename_dict = {}
    if lifetime_col:
        new_name = f"{lifetime_col} [years]"
        rename_dict[lifetime_col] = new_name
        lifetime_col = new_name
    if rad_eff_col:
        new_name = f"{rad_eff_col} [W m-2 ppb-1]"
        rename_dict[rad_eff_col] = new_name
        rad_eff_col = new_name
    
    if rename_dict:
        df_ar6.rename(columns=rename_dict, inplace=True)
    
    # Normalize CAS numbers
    df_ar6[cas_col] = df_ar6[cas_col].apply(normalize_cas)
    

    column_names = {
        'name': name_col,
        'cas': cas_col,
        'lifetime': lifetime_col,
        'rad_eff': rad_eff_col,
        'formula': formula_col
    }
    
    return df_ar6, column_names


def fetch_hodnebrog_data() -> tuple:
    """
    Fetch Hodnebrog et al. (2020) GHG properties from GitHub
    
    Returns:
        Tuple of (dataframe, column_names_dict)
        - dataframe: DataFrame with Hodnebrog data
        - column_names: Dict with keys: 'cas', 'rad_eff', 'formula'
    """

    
    hodne_url = "https://raw.githubusercontent.com/chrisroadmap/ar6/main/data_input/Hodnebrog_et_al_2020_revgeo/hodnebrog20.csv"
    df_hodne = pd.read_csv(hodne_url)
    
    # Detect column names
    hod_cas_col = detect_column(df_hodne, ['cas'])
    hod_rad_col = detect_column(df_hodne, ['re', 'radiative'])
    hod_formula_col = detect_column(df_hodne, ['formula'])
    
    if hod_cas_col is None:
        raise ValueError("CAS column not found in Hodnebrog table")
    
    # Normalize CAS numbers
    df_hodne[hod_cas_col] = df_hodne[hod_cas_col].apply(normalize_cas)
    
    column_names = {
        'cas': hod_cas_col,
        'rad_eff': hod_rad_col,
        'formula': hod_formula_col
    }
    
    return df_hodne, column_names


"""
Module for data enrichment and augmentation
"""

MANUAL_GHG_DATA = {
    "Carbon dioxide": {"cas": "124-38-9", "rad_eff": 1.33e-5},
    "Methane": {"cas": "74-82-8", "rad_eff": 0.000388},
    "Nitrous oxide": {"cas": "10024-97-2", "rad_eff": 0.0032},
}


def enrich_ar6_with_manual_ghg(df_ar6: pd.DataFrame, column_names: dict) -> pd.DataFrame:
    """    
    Enrich AR6 data manually with more precise radiative efficiency values for CO2, CH$ and N2O
    
    Args:
        df_ar6: AR6 dataframe
        column_names: Dict with column name mappings
        
    Returns:
        Updated dataframe
    """
    
    name_col = column_names['name']
    cas_col = column_names['cas']
    rad_eff_col = column_names['rad_eff']
    
    for substance_name, info in MANUAL_GHG_DATA.items():
        cas_number = info["cas"]
        rad_val = info["rad_eff"]
        
        # Find matching row by name
        for idx, row in df_ar6.iterrows():
            if name_col and pd.notna(row[name_col]):
                if str(row[name_col]).strip().lower() == substance_name.lower():
                    df_ar6.loc[idx, cas_col] = normalize_cas(cas_number)
                    df_ar6.loc[idx, rad_eff_col] = rad_val
    
    return df_ar6


def enrich_ar6_with_hodnebrog(df_ar6: pd.DataFrame, df_hodne: pd.DataFrame,
                              ar6_cols: dict, hodne_cols: dict) -> tuple:
    """
    Enrich AR6 data with Hodnebrog et al. (2020) values
    
    Attempts CAS-based matching first, then formula-based matching
    
    Args:
        df_ar6: AR6 dataframe
        df_hodne: Hodnebrog dataframe
        ar6_cols: AR6 column name mappings
        hodne_cols: Hodnebrog column name mappings
        
    Returns:
        Tuple of (updated_df_ar6, update_counts)
    """
    
    cas_col = ar6_cols['cas']
    rad_eff_col = ar6_cols['rad_eff']
    formula_col = ar6_cols['formula']
    
    hod_cas_col = hodne_cols['cas']
    hod_rad_col = hodne_cols['rad_eff']
    hod_formula_col = hodne_cols['formula']
    
    if not (cas_col and rad_eff_col and hod_cas_col and hod_rad_col):
        return df_ar6, {'rad_eff_updates': 0}
    
    cas_series = df_ar6[cas_col]
    cas_valid = cas_series.notna() & ~cas_series.isin(["0", 0])
    
    # Build CAS -> radiative efficiency map (first non-null value wins)
    hodne_cas = df_hodne[[hod_cas_col, hod_rad_col]].copy()
    hodne_cas[hod_rad_col] = pd.to_numeric(hodne_cas[hod_rad_col], errors="coerce")
    hodne_cas = hodne_cas.dropna(subset=[hod_cas_col, hod_rad_col])
    cas_rad_map = hodne_cas.drop_duplicates(subset=[hod_cas_col]).set_index(hod_cas_col)[hod_rad_col]
    
    new_rad = pd.Series(index=df_ar6.index, dtype="float64")
    cas_rad = cas_series.map(cas_rad_map)
    new_rad.loc[cas_valid] = cas_rad.loc[cas_valid]
    
    # Formula-based matching only for invalid CAS
    if formula_col and hod_formula_col:
        ar6_formula_norm = df_ar6[formula_col].astype(str).str.strip().str.lower()
        hodne_formula = df_hodne[[hod_formula_col, hod_rad_col]].copy()
        hodne_formula[hod_rad_col] = pd.to_numeric(hodne_formula[hod_rad_col], errors="coerce")
        hodne_formula[hod_formula_col] = hodne_formula[hod_formula_col].astype(str).str.strip().str.lower()
        hodne_formula = hodne_formula.dropna(subset=[hod_formula_col, hod_rad_col])
        formula_rad_map = hodne_formula.drop_duplicates(subset=[hod_formula_col]).set_index(hod_formula_col)[hod_rad_col]
        
        formula_rad = ar6_formula_norm.map(formula_rad_map)
        cas_invalid = ~cas_valid
        new_rad.loc[cas_invalid] = new_rad.loc[cas_invalid].combine_first(formula_rad.loc[cas_invalid])
    
    update_mask = new_rad.notna()
    df_ar6.loc[update_mask, rad_eff_col] = new_rad.loc[update_mask]
    
    return df_ar6, {'rad_eff_updates': int(update_mask.sum())}

