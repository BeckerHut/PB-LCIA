import pandas as pd
import numpy as np
import bw2data as bd
from bw2io import create_default_biosphere3
from bw2data import databases

"""
Module for extracting and formatting biosphere database flows
"""

def get_biosphere_db(database_name: str = "biosphere3") -> pd.DataFrame:
    """
    Ensure biosphere database exists (create if needed) and extract all flows into a DataFrame.
    
    Args:
        database_name: Name of the biosphere database (default: "biosphere3")
        
    Returns:
        DataFrame with biosphere flows and their properties
    """
    
    # Check if database exists, create if not
    if database_name not in databases:
        if database_name == "biosphere3":
            create_default_biosphere3()
            print(f"{database_name} created successfully.")
        else:
            raise ValueError(f"Database {database_name} not found and cannot be auto-created.")
    
    # Extract flows from the database
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

"""
Module for creating a dummy database
"""

def create_dummy_db(num_flows: int = 10, 
                    start_year: int = 2020, 
                    years: int = 3,) -> pd.DataFrame:
    """
    Create a dummy database with randomly selected biosphere flows.
    Each flow gets assigned a random emission amount between 0 and 5 for each year.
    
    Args:
        num_flows: Number of random flows to select (default: 10)
        start_year: Starting year for temporal data (default: 2020)
        years: Number of years to create data for (default: 3)
    Returns:
        DataFrame with randomly selected biosphere flows and temporal resolution
    """
    # Get biosphere database
    df = get_biosphere_db()
    
    # Select random flows (different selection each time)
    random_flows = df.sample(n=num_flows)

    # Create temporal data: for each flow with random amounts between 0 and 5
    temporal_data = []

    for _, row in random_flows.iterrows():
        for year in range(years):
            date = f"{start_year + year}-01-01"
            amount = np.random.randint(0, 6)  # Random integer between 0 and 5 (inclusive)
            
            temporal_row = {
                'date': date,
                'amount': amount,
                'activity': 1,
                'name': row.get('name', ''),
                'categories': row.get('categories', ''),
                'unit': row.get('unit', ''),
                'CAS': row.get('CAS', ''),
            }
            temporal_data.append(temporal_row)

    # Create the output dataframe
    df_dummy = pd.DataFrame(temporal_data)
    df_dummy['date'] = pd.to_datetime(df_dummy['date'])
    
    return df_dummy