"""
Module for fetching external data sources (AR6, Hodnebrog)
"""

import pandas as pd
import requests
from io import StringIO
from filters import normalize_cas


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
    
    print(f"Total rows in CSV: {len(df_ar6)}")
    
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
    
    # Add molar mass column
    mol_col = "Molar_mass [kg mol-1]"
    df_ar6[mol_col] = pd.NA
    
    # Normalize CAS numbers
    df_ar6[cas_col] = df_ar6[cas_col].apply(normalize_cas)
    

    column_names = {
        'name': name_col,
        'cas': cas_col,
        'lifetime': lifetime_col,
        'rad_eff': rad_eff_col,
        'formula': formula_col,
        'molar_mass': mol_col
    }
    
    return df_ar6, column_names


def fetch_hodnebrog_data() -> tuple:
    """
    Fetch Hodnebrog et al. (2020) GHG properties from GitHub
    
    Returns:
        Tuple of (dataframe, column_names_dict)
        - dataframe: DataFrame with Hodnebrog data
        - column_names: Dict with keys: 'cas', 'rad_eff', 'molar_mass', 'formula'
    """

    
    hodne_url = "https://raw.githubusercontent.com/chrisroadmap/ar6/main/data_input/Hodnebrog_et_al_2020_revgeo/hodnebrog20.csv"
    df_hodne = pd.read_csv(hodne_url)
    
    # Detect column names
    hod_cas_col = detect_column(df_hodne, ['cas'])
    hod_rad_col = detect_column(df_hodne, ['re', 'radiative'])
    hod_mol_col = detect_column(df_hodne, ['molar'])
    hod_formula_col = detect_column(df_hodne, ['formula'])
    
    if hod_cas_col is None:
        raise ValueError("CAS column not found in Hodnebrog table")
    
    # Normalize CAS numbers
    df_hodne[hod_cas_col] = df_hodne[hod_cas_col].apply(normalize_cas)
    
    column_names = {
        'cas': hod_cas_col,
        'rad_eff': hod_rad_col,
        'molar_mass': hod_mol_col,
        'formula': hod_formula_col
    }
    
    return df_hodne, column_names
