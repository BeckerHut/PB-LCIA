"""
Module for extracting and formatting biosphere database flows
"""

import pandas as pd
import bw2data as bd


def extract_biosphere_flows(database_name: str = "biosphere3") -> pd.DataFrame:
    """
    Extract all flows from a biosphere database into a DataFrame
    
    Args:
        database_name: Name of the biosphere database (default: "biosphere3")
        
    Returns:
        DataFrame with biosphere flows and their properties
    """
    bio = bd.Database(database_name)
    
    rows = []
    for flow in bio:
        rows.append({
            "name": flow.get("name"),
            "categories": flow.get("categories"),  # compartment
            "unit": flow.get("unit"),
            "type": flow.get("type"),  # e.g., "emission", "resource"
            "location": flow.get("location"),
            "database": flow.get("database"),
            "code": flow.get("code"),
            "reference_product": flow.get("reference product"),
            "comment": flow.get("comment"),
            "CAS": flow.get("CAS number"),
            "formula": flow.get("formula"),
        })
    
    df = pd.DataFrame(rows)
    
    # Sort for readability
    df = df.sort_values(["categories", "name"], na_position="last").reset_index(drop=True)
    
    return df


def ensure_database_exists(database_name: str = "biosphere3") -> bool:
    """
    Check if database exists, create if not
    
    Args:
        database_name: Name of the database
        
    Returns:
        True if database exists or was created, False otherwise
    """
    from bw2io import create_default_biosphere3
    from bw2data import databases
    
    if database_name not in databases:
        if database_name == "biosphere3":
            create_default_biosphere3()
            print(f"{database_name} created.")
            return True
        else:
            print(f"Database {database_name} not found and cannot be auto-created.")
            return False
    else:
        print(f"{database_name} already exists in current project.")
        return True
