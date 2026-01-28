from bw2data import projects, databases
import pandas as pd
import numpy as np
from biosphere_export import ensure_database_exists, extract_biosphere_flows

def make_dummy_database(
    num_flows: int = 10,
    start_year: int = 2020,
    years: int = 5,
    amount_per_year: float = 1.0,
    activity_id: int = 1,
):
    """
    Returns:
      df: DataFrame with columns date, amount, flow, activity (+ optional metadata cols)
      mapping: dict of requested flow labels -> resolved biosphere3 flow integer ids
    """
    # Set the current project
    projects.set_current("PB_LCIA")

    # Create or load biosphere3 database

    ensure_database_exists("biosphere3")

    # Export biosphere flows to DataFrame

    df = extract_biosphere_flows("biosphere3")
    print(f"\nTotal biosphere flows extracted: {len(df)}")
    

    # Select 10 random flows from df
    np.random.seed(32)
    random_flows = df.sample(n=num_flows, random_state=42)

    # Create temporal data: 3 years for each flow with random amounts between 0 and 5
    years = 3
    start_year = 2020
    temporal_data = []

    for _, row in random_flows.iterrows():
        for year in range(years):
            date = f"{start_year + year}-01-01"
            amount = np.random.randint(0, 6)  # Random integer between 0 and 5 (inclusive)
            
            # Extract flow name - try multiple possible column names
            flow_name = row.get('flow_name', row.get('name', ''))
            flow_label = flow_name[:20] if pd.notna(flow_name) and len(str(flow_name)) > 0 else ''
            
            temporal_row = {
                'date': date,
                'amount': amount,
                'activity': 1,
                'flow_name': flow_name,
                'categories': row.get('categories', ''),
                'unit': row.get('unit', ''),
            }
            temporal_data.append(temporal_row)

    # Create the output dataframe
    df_temporal = pd.DataFrame(temporal_data)
    df_temporal['date'] = pd.to_datetime(df_temporal['date'])

    print(f"\nRandom {num_flows} flows of biosphere3 with temporal data:")
    print(df_temporal)
    
    
    return df_temporal