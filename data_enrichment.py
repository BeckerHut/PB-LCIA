"""
Module for data enrichment and augmentation
"""

import pandas as pd
from filters import normalize_cas


MANUAL_GHG_DATA = {
    "Carbon dioxide": {"cas": "124-38-9", "rad_eff": 1.33e-5, "m_mol": 0.04401},
    "Methane": {"cas": "74-82-8", "rad_eff": 0.000388, "m_mol": 0.01604},
    "Nitrous oxide": {"cas": "10024-97-2", "rad_eff": 0.0032, "m_mol": 0.04401},
}


def enrich_ar6_with_manual_ghg(df_ar6: pd.DataFrame, column_names: dict) -> pd.DataFrame:
    """
    Enrich AR6 data with manually corrected GHG entries (CO2, CH4, N2O)
    
    Uses more precise radiative efficiency and molar mass values for key GHGs
    
    Args:
        df_ar6: AR6 dataframe
        column_names: Dict with column name mappings
        
    Returns:
        Updated dataframe
    """
    
    name_col = column_names['name']
    cas_col = column_names['cas']
    rad_eff_col = column_names['rad_eff']
    mol_col = column_names['molar_mass']
    
    for substance_name, info in MANUAL_GHG_DATA.items():
        cas_number = info["cas"]
        rad_val = info["rad_eff"]
        m_mol = info["m_mol"]
        
        # Find matching row by name
        for idx, row in df_ar6.iterrows():
            if name_col and pd.notna(row[name_col]):
                if str(row[name_col]).strip().lower() == substance_name.lower():
                    df_ar6.loc[idx, cas_col] = normalize_cas(cas_number)
                    df_ar6.loc[idx, rad_eff_col] = rad_val
                    df_ar6.loc[idx, mol_col] = m_mol
                    print(f"  Updated '{substance_name}' -> CAS {cas_number}, "
                          f"Radiative_Efficiency {rad_val}, Molar_Mass {m_mol}")
    
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
    mol_col = ar6_cols['molar_mass']
    formula_col = ar6_cols['formula']
    
    hod_cas_col = hodne_cols['cas']
    hod_rad_col = hodne_cols['rad_eff']
    hod_mol_col = hodne_cols['molar_mass']
    hod_formula_col = hodne_cols['formula']
    
    updates = 0
    mm_updates = 0
    
    for idx, row in df_ar6.iterrows():
        cas_value = row[cas_col]
        matched_row = None
        
        # Try CAS-based matching first
        if pd.notna(cas_value) and cas_value not in ("0", 0):
            cas_matches = df_hodne[df_hodne[hod_cas_col] == cas_value]
            if not cas_matches.empty:
                matched_row = cas_matches.iloc[0]
        
        # Try formula-based matching if CAS is invalid
        if matched_row is None and cas_value in ("0", 0) and formula_col and hod_formula_col and pd.notna(row[formula_col]):
            target_formula = str(row[formula_col]).strip().lower()
            formula_matches = df_hodne[df_hodne[hod_formula_col].astype(str).str.strip().str.lower() == target_formula]
            if not formula_matches.empty:
                matched_row = formula_matches.iloc[0]
        
        # Update values from matched row
        if matched_row is not None:
            if hod_rad_col and pd.notna(matched_row[hod_rad_col]):
                rad_val = pd.to_numeric(matched_row[hod_rad_col], errors="coerce")
                if pd.notna(rad_val):
                    df_ar6.loc[idx, rad_eff_col] = rad_val
                    updates += 1
            
            if hod_mol_col and pd.notna(matched_row[hod_mol_col]):
                mm_val = pd.to_numeric(matched_row[hod_mol_col], errors="coerce")
                if pd.notna(mm_val):
                    df_ar6.loc[idx, mol_col] = mm_val
                    mm_updates += 1
    
    print(f"Updated radiative efficiency from Hodnebrog for {updates} rows.")
    print(f"Updated molar mass from Hodnebrog for {mm_updates} rows.")
    
    df_no_molar_mass = df_ar6[df_ar6[mol_col].isna()]
    print(f"\nRemaining entities with no molar mass: {len(df_no_molar_mass)}")
    
    return df_ar6, {'rad_eff_updates': updates, 'molar_mass_updates': mm_updates}
