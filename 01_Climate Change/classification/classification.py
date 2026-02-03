import pandas as pd
from .filters import (
    CompartmentFilter, ExcludeCompartmentFilter, CASNumberFilter, 
    SpecificCASFilter, VOCFilter, NameFilter, apply_filter, 
    normalize_cas, count_carbon_atoms
)
from .flow_matching import match_flows

"""
Module for classifying relevant elementary flows for Climate Change characterization
"""

def classify(df, df_ar6, ar6_cols, online_lookup=False):
    """
    Classify air compartment flows into VOC and GHG categories.
    
    Args:
        df: Biosphere flows dataframe
        df_ar6: AR6 GHG reference dataframe
        ar6_cols: Column name mappings for AR6 dataframe
        online_lookup: If True, use PubChem online lookup for classification enrichment (default: False)
    
    Returns:
        Tuple of (df_voc_lf, df_voc_no_lifetime, df_other_ghg, df_no_ghg, df_unknown)
        - df_voc_lf: VOC flows with AR6 lifetime data
        - df_voc_no_lifetime: VOC flows without lifetime (includes unspecified VOCs)
        - df_other_ghg: NOT_VOC and UNKNOWN flows matched to AR6
        - df_no_ghg: NOT_VOC flows with no AR6 match
        - df_unknown: UNKNOWN flows with no AR6 match
    """
    
    # Step 1: Filter for air compartment, excluding natural resource, soil, and water
    df_air = apply_filter(df, CompartmentFilter("air"))
    df_air = apply_filter(df_air, ExcludeCompartmentFilter(["natural resource", "soil", "water"]))
    
    # Step 2: Separate CO2 from other flows
    df_air_reset = df_air.reset_index().rename(columns={'index': 'orig_index'})
    cas_co2 = ["124-38-9"]
    df_co2_raw = apply_filter(df_air_reset, SpecificCASFilter(cas_co2))
    matched_idx = df_co2_raw['orig_index']
    
    df_nonco2 = (
        df_air_reset.loc[~df_air_reset['orig_index'].isin(matched_idx)]
        .drop(columns=['orig_index'])
        .reset_index(drop=True)
    )
    
    df_co2 = (
        df_co2_raw.drop(columns=['orig_index'])
        .reset_index(drop=True)
    )
    
    
    # Step 3: Split by CAS presence
    df_cas = apply_filter(df_nonco2, CASNumberFilter(cas=True))
    df_no_cas = apply_filter(df_nonco2, CASNumberFilter(cas=False))
    
    
    # Step 4: Classify flows as VOC/NOT_VOC/UNKNOWN

    # Normalize CAS numbers for classification
    df_cas = df_cas.reset_index(drop=True)
    df_cas["CAS"] = df_cas["CAS"].apply(normalize_cas)
    
    clf = VOCFilter("UNKNOWN", online_lookup=online_lookup, cache_db="voc_cache.sqlite")
    
    def classify_with_details(row):
        cas = normalize_cas(row.get("CAS"))
        flow_name = row.get("name")
        status = clf._classify_flow(cas, flow_name=flow_name)
        entry = clf._classification_cache.get(cas, {})
        return pd.Series({
            "formula": entry.get("formula"),
            "carbon_atoms": count_carbon_atoms(entry.get("formula")),
            "bp_c": entry.get("bp_c"),
            "voc_status": status,
            "source": entry.get("source"),
            "Molar mass [g mol-1]": entry.get("molar_mass"),
        })
    
    details = df_cas.apply(classify_with_details, axis=1)
    for col in details.columns:
        df_cas[col] = details[col].values
    
    # Split into VOC/NOT_VOC/UNKNOWN
    df_voc = df_cas[df_cas["voc_status"] == "VOC"]
    df_not_voc = df_cas[df_cas["voc_status"] == "NOT_VOC"]
    df_unknown = df_cas[df_cas["voc_status"] == "UNKNOWN"]
    
    if online_lookup == True:
        print("\nOnline failure messages (count):")
        mask_fail = df_cas["source"].fillna("").str.startswith("online lookup failed")
        print(df_cas.loc[mask_fail, "source"].value_counts().to_string())
    
    
    # Step 5: Match flows against AR6 table 

    # Find CAS column name of AR6 data
    flow_cas_col = None
    for col in df_cas.columns:
        if 'cas' in col.lower():
            flow_cas_col = col
            break
    
    # Match VOC flows against AR6 table
    matched_voc, not_matched_voc = match_flows(
        df_voc,
        df_ar6,
        flow_cas_col=flow_cas_col,
        ar6_cas_col=ar6_cols['cas'],
        ar6_cols={
            'lifetime': ar6_cols['lifetime'],
            'rad_eff': ar6_cols['rad_eff'],
        }
    )
    
    # Match NOT_VOC flows against AR6 table
    matched_not_voc, not_matched_not_voc = match_flows(
        df_not_voc,
        df_ar6,
        flow_cas_col=flow_cas_col,
        ar6_cas_col=ar6_cols['cas'],
        ar6_cols={
            'lifetime': ar6_cols['lifetime'],
            'rad_eff': ar6_cols['rad_eff'],
        }
    )
    # Match UNKNOWN flows against AR6 table
    matched_unknown, not_matched_unknown = match_flows(
        df_unknown,
        df_ar6,
        flow_cas_col=flow_cas_col,
        ar6_cas_col=ar6_cols['cas'],
        ar6_cols={
            'lifetime': ar6_cols['lifetime'],
            'rad_eff': ar6_cols['rad_eff'],
        }
    )
    
    # Step 7: Organize results
    df_voc_lt = matched_voc
    df_voc_nolt = not_matched_voc.reset_index(drop=True)
    
    df_other_ghg = pd.concat([matched_not_voc, matched_unknown], ignore_index=True)
    
    df_no_ghg = not_matched_not_voc.reset_index(drop=True)
    df_unknown_remaining = not_matched_unknown.reset_index(drop=True)
    
    # Step 8: Filter unspecified VOCs from no CAS list 
    unspecified_voc = ["VOC", "Aldehydes", "Hydrocarbons", "NMVOC"]
    df_voc_unspecified = apply_filter(df_no_cas, NameFilter(unspecified_voc))


    
    
    print(f"Total flows in 'air' compartment: {len(df_air)}")
    print(f"|- Flows with CO2: {len(df_co2)}")
    print(f"|- Flows excluding CO2: {len(df_nonco2)}")
    print(f"   |- Non-CO2 flows WITH CAS: {len(df_cas)}")
    print(f"      |- VOC flows with lifetime: {len(df_voc_lt)}")
    print(f"      |- VOC flows without lifetime: {len(df_voc_nolt)}")
    print(f"      |- Other GHG flows: {len(df_other_ghg)}")
    print(f"      |- Non GHG flows: {len(df_no_ghg)}")
    print(f"      |- Still unknown flows: {len(df_unknown_remaining)}")
    print(f"   |- Non-CO2 flows WITHOUT CAS: {len(df_no_cas)}")
    print(f"      |- Unspecified VOC flows: {len(df_voc_unspecified)}")
    
    
    return df_co2, df_voc_lt, df_voc_nolt, df_other_ghg, df_no_ghg, df_unknown_remaining, df_voc_unspecified