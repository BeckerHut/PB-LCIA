"""
Module for matching flows with CSV/AR6 data
"""

import pandas as pd


def match_flows(df_with_cas: pd.DataFrame, df_ar6: pd.DataFrame,
                flow_cas_col: str, ar6_cas_col: str,
                ar6_cols: dict) -> tuple:
    """
    Match flows (with CAS numbers) to AR6 reference data and enrich with properties
    
    Args:
        df_with_cas: DataFrame of flows that have CAS numbers
        df_ar6: AR6 reference data
        flow_cas_col: Column name for CAS in the flows
        ar6_cas_col: Column name for CAS in AR6 data
        ar6_cols: Dict with AR6 column mappings (lifetime, rad_eff, molar_mass)
        
    Returns:
        Tuple of (df_matched, df_not_matched)
    """
    
    lifetime_col = ar6_cols['lifetime']
    rad_eff_col = ar6_cols['rad_eff']
    mol_col = ar6_cols['molar_mass']
    
    matching_flows = []
    non_matching_flows = []
    
    for idx, row in df_with_cas.iterrows():
        cas_flow = row[flow_cas_col]
        
        # Find if this CAS matches any in the CSV
        csv_match = df_ar6[df_ar6[ar6_cas_col] == cas_flow]
        
        if not csv_match.empty:
            # Found a match
            match_row = csv_match.iloc[0]
            flow_data = row.to_dict()
            flow_data['Lifetime [years]'] = pd.to_numeric(match_row.get(lifetime_col), errors='coerce') if lifetime_col else None
            flow_data['Radiative_Efficiency [W m-2 ppb-1]'] = pd.to_numeric(match_row.get(rad_eff_col), errors='coerce') if rad_eff_col else None
            flow_data['Molar_Mass [kg mol-1]'] = pd.to_numeric(match_row.get(mol_col), errors='coerce') if mol_col else None
            matching_flows.append(flow_data)
        else:
            # No match found
            non_matching_flows.append(row)
    
    # Create output dataframes
    df_matched = pd.DataFrame(matching_flows) if matching_flows else pd.DataFrame()
    df_not_matched = pd.DataFrame(non_matching_flows) if non_matching_flows else pd.DataFrame()
    
    
    
    return df_matched, df_not_matched