import pandas as pd
from abc import ABC, abstractmethod

# Define modular filter rules
# Abstract Base Class allows creating a class 
# that can't be instantiated directly, only inherited from

class FilterRule(ABC):
    """Base class for filter rules"""
    
    @abstractmethod 
    # this decorator means any subclass must implement this method
    # makes code modular
    def apply(self, flow: dict) -> bool:
        """Return True if flow passes the filter, False otherwise"""
        pass


class CompartmentFilter(FilterRule):
    """Filter flows by compartment (e.g., 'air', 'water', 'soil')"""
    
    def __init__(self, compartment: str):
        self.compartment = compartment.lower() 
        # converts compartment to lowercase for case insensitive comparison
        # and stores it so it can be used in apply method
    
    def apply(self, flow: dict) -> bool:
        categories = flow.get("categories", [])
        # compartments are called categories in the biosphere flows
        if not categories:
            return False
        # if there are no compartments return false
        return any(self.compartment in cat.lower() for cat in categories)
        # checks if the compartment is in any of the categories


class CASNumberFilter(FilterRule):
    """Filter flows that have (or don't have) a CAS number"""
    
    def __init__(self, cas: bool = True):
        # if cas is True, filter for flows that have a CAS number
        # if cas is False, filter for flows that do not have a CAS number
        self.cas = cas
    
    def apply(self, flow: dict) -> bool:
        casrn = flow.get("CAS")
        if self.cas:
            return casrn is not None and casrn != ""
        # case when looking for CAS
        # returns true if casrn is not None and not empty
        else:
            return casrn is None or casrn == ""
        # case when looking for no CAS
        # returns true if casrn is None or empty

class NameFilter(FilterRule):
    """Filter flows by name substring (case-insensitive)"""
    
    def __init__(self, substring: str):
        self.substring = substring.lower()
    
    def apply(self, flow: dict) -> bool:
        name = flow.get("name", "").lower()
        return self.substring in name
    # checks if the substring is in the name    


def apply_filter(df: pd.DataFrame, filter_rule: FilterRule) -> pd.DataFrame:
    """Apply a single filter rule to the dataframe and return filtered results"""
    # Convert dataframe rows to dict format for filtering
    filtered_rows = []
    for idx, row in df.iterrows():
        # Get categories (could be tuple or list from biosphere)
        categories = row["categories"]
        if isinstance(categories, (list, tuple)):
            categories_list = list(categories)
        else:
            categories_list = []
        
        flow_dict = {
            "name": row["name"],
            "categories": categories_list,
            "CAS": row["CAS"],
            # Add other fields as needed
        }
        if filter_rule.apply(flow_dict):
            filtered_rows.append(idx)
    
    return df.loc[filtered_rows].reset_index(drop=True)


def normalize_cas(cas_string):
    """Strip leading zeros from CAS number (e.g., '007732-18-5' -> '7732-18-5')"""
    if pd.isna(cas_string) or cas_string == "":
        return None
    cas_str = str(cas_string).strip()
    parts = cas_str.split('-')
    if parts:
        parts[0] = parts[0].lstrip('0') or '0'
    return '-'.join(parts)
