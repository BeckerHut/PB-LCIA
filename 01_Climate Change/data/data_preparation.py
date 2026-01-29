import pandas as pd
import requests
from io import StringIO
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

