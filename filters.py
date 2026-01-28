import pandas as pd # for dataframe operations
import re # for regex matching pattern 
import sqlite3 # for local database caching
import datetime as dt # for adding timestemps
import requests # for http requests
from abc import ABC, abstractmethod # for creating an abstract base class
from typing import Optional, Tuple, Dict, Any # for improving code readability
from contextlib import closing # for context-managed SQLite connections

# Import chemicals library (required for NMVOC classification)
try:
    from chemicals.phase_change import Tb, Tb_methods
    from chemicals.identifiers import MW
    from chemicals import search_chemical
    CHEMICALS_AVAILABLE = True
except ImportError as e:
    CHEMICALS_AVAILABLE = False
    IMPORT_ERROR = str(e)


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

class ExcludeCompartmentFilter(FilterRule):
    """Exclude flows that have any of the specified compartments"""
    
    def __init__(self, compartments: list):
        # Store compartments as lowercase for case-insensitive comparison
        self.compartments = [comp.lower() for comp in compartments]
    
    def apply(self, flow: dict) -> bool:
        categories = flow.get("categories", [])
        # compartments are called categories in the biosphere flows
        # Return True (keep) if NONE of the excluded compartments are in categories
        return not any(
            excluded_comp in cat.lower() 
            for cat in categories 
            for excluded_comp in self.compartments
        )

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


class SpecificCASFilter(FilterRule):
    """Filter flows by specific CAS number(s)"""
    
    def __init__(self, cas_numbers: list):
        # Normalize CAS numbers on initialization
        self.cas_numbers = [normalize_cas(cas) for cas in cas_numbers]
    
    def apply(self, flow: dict) -> bool:
        casrn = normalize_cas(flow.get("CAS"))
        return casrn in self.cas_numbers


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
    # check if CAS exists
    if pd.isna(cas_string) or cas_string == "":
        return None
    cas_str = str(cas_string).strip()
    parts = cas_str.split('-')
    # strip away leading zeros
    if parts:
        parts[0] = parts[0].lstrip('0') or '0'
    return '-'.join(parts)


# ============================================================================
# NMVOC Classification Filter
# ============================================================================

# Configuration
DEFAULT_THRESHOLD_C = 250.0

OFFLINE_METHOD_PRIORITY = [
    "CRC_ORG",
    "CRC_INORG",
    "YAWS",
]

EXCLUDE_METHODS_BY_DEFAULT = {
    "WIKIDATA",
    "WEBBOOK",
    "COMMON_CHEMISTRY",
    "HEOS",
}



# SQLite cache helpers

def init_cache(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite cache for NMVOC classification results."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            cas TEXT PRIMARY KEY,
            name TEXT,
            bp_c REAL,
            formula TEXT,
            molar_mass_g_mol REAL,
            source TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, cas: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached NMVOC classification result."""
    cur = conn.execute(
        "SELECT cas, name, bp_c, formula, molar_mass_g_mol, source, updated_at FROM results WHERE cas=?",
        (cas,),
    )
    row = cur.fetchone()
    if not row:
        return None

    return {
        "cas": row[0],
        "name": row[1],
        "bp_c": row[2],
        "formula": row[3],
        "molar_mass": row[4],
        "source": row[5],
        "updated_at": row[6],
    }


def cache_put(
    conn: sqlite3.Connection,
    cas: str,
    name: str,
    bp_c: Optional[float],
    formula: Optional[str],
    molar_mass_g_mol: Optional[float] = None,
    source: str = "",
):
    """Store NMVOC classification result in cache."""
    conn.execute(
        "INSERT OR REPLACE INTO results (cas, name, bp_c, formula, molar_mass_g_mol, source, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (cas, name, bp_c, formula, molar_mass_g_mol, source, dt.datetime.now(dt.timezone.utc).isoformat()),
    )
    conn.commit()


# Offline lookups

def offline_bp_lookup_c(cas: str, allow_estimates: bool = False) -> Tuple[Optional[float], str]:
    """Look up boiling point using the chemicals library (offline)."""
    if not CHEMICALS_AVAILABLE:
        return None, "chemicals library not installed"

    cas = str(cas).strip()
    methods = Tb_methods(cas)
    if not methods:
        return None, "no Tb methods available"

    filtered = [m for m in methods if m not in EXCLUDE_METHODS_BY_DEFAULT]
    if allow_estimates and "JOBACK" in methods and "JOBACK" not in filtered:
        filtered.append("JOBACK")

    chosen = None
    for pref in OFFLINE_METHOD_PRIORITY + (["JOBACK"] if allow_estimates else []):
        if pref in filtered:
            chosen = pref
            break
    if chosen is None and filtered:
        chosen = filtered[0]
    if chosen is None:
        return None, "no suitable Tb method"

    try:
        bp_k = Tb(cas, method=chosen)
    except Exception as e:
        return None, f"Tb failed: {type(e).__name__}"

    if bp_k is None:
        return None, "Tb returned None"

    return float(bp_k - 273.15), chosen


def offline_formula_lookup(cas: str) -> Tuple[Optional[str], str]:
    """Look up molecular formula using the chemicals library (offline)."""
    if not CHEMICALS_AVAILABLE:
        return None, "chemicals library not installed"

    cas = str(cas).strip()
    try:
        result = search_chemical(cas)
        if result and hasattr(result, 'formula') and result.formula:
            return str(result.formula).strip(), "chemicals"
    except Exception as e:
        return None, f"formula lookup failed: {type(e).__name__}"

    return None, "formula not available offline"


def offline_molar_mass_lookup(cas: str) -> Tuple[Optional[float], str]:
    """Look up molar mass using the chemicals library (offline)."""
    if not CHEMICALS_AVAILABLE:
        return None, "chemicals library not installed"

    cas = str(cas).strip()
    try:
        mw = MW(cas)
        if mw is not None and mw > 0:
            return float(mw), "chemicals"
    except Exception:
        pass

    return None, "molar mass not available offline"


def has_carbon(formula: Optional[str]) -> bool:
    """Check if formula contains real carbon (C not as Cl, Ca, Cd, etc.)
    
    Args:
        formula: Molecular formula string (e.g., "C6H14", "NaCl", "H2O")
    
    Returns:
        True if formula contains at least one carbon atom, False otherwise
    """
    if not formula:
        return False
    
    formula = str(formula).strip()
    match = re.search(r'C(?![a-z])', formula)
    return match is not None


def classify_nmvoc(
    bp_c: Optional[float],
    has_formula: bool,
    is_organic: bool,
    threshold_c: float = DEFAULT_THRESHOLD_C,
) -> Tuple[str, str]:
    """Classify compound as NMVOC, NOT_NMVOC, or UNKNOWN.
    
    Logic:
    - NMVOC: organic AND bp < 250°C
    - NOT_NMVOC: bp >= 250°C OR (formula exists AND not organic)
    - UNKNOWN: formula unavailable AND (bp unavailable OR bp < 250°C)
    """
    # Case 1: BP >= 250°C -> NOT_NMVOC
    if bp_c is not None and bp_c >= threshold_c:
        return "NOT_NMVOC", f"BP {bp_c:.2f}°C >= {threshold_c}°C"
    
    # Case 2: Formula available and not organic -> NOT_NMVOC
    if has_formula and not is_organic:
        return "NOT_NMVOC", "Not organic (no carbon in formula)"
    
    # Case 3: Formula available AND organic AND BP < 250°C -> NMVOC
    if has_formula and is_organic and bp_c is not None and bp_c < threshold_c:
        return "NMVOC", f"Organic + BP {bp_c:.2f}°C < {threshold_c}°C"
    
    # Case 4: No formula -> UNKNOWN
    if not has_formula:
        if bp_c is not None and bp_c < threshold_c:
            return "UNKNOWN", "Formula unavailable, BP < 250°C"
        else:
            return "UNKNOWN", "Formula unavailable"
    
    # Case 5: Formula exists, is organic, but no BP -> UNKNOWN
    if has_formula and is_organic and bp_c is None:
        return "UNKNOWN", "Organic but BP unavailable"
    
    # Fallback
    return "UNKNOWN", "Unable to classify"


# CAS validation

CAS_REGEX = re.compile(r"^\d{2,7}-\d{2}-\d$")

def cas_checksum_ok(cas: str) -> bool:
    """Validate CAS checksum."""
    parts = cas.split("-")
    if len(parts) != 3:
        return False
    try:
        digits = "".join(parts[0:2])
        checksum = int(parts[2])
        total = sum(int(d) * (i + 1) for i, d in enumerate(reversed(digits)))
        return total % 10 == checksum
    except Exception:
        return False


def is_valid_cas(cas: str) -> bool:
    """Check if CAS number is valid format."""
    if not cas:
        return False
    cas = cas.strip()
    if not CAS_REGEX.match(cas):
        return False
    return cas_checksum_ok(cas)


# Online lookup helpers

def pubchem_fetch(cas: str, timeout: float = 8.0) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Fetch formula, molar mass, and boiling point from PubChem REST API.
    
    Args:
        cas: CAS number string
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (formula, molar_mass_g_mol, bp_c) where any may be None
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/JSON"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None, None, None

    def find_sections(sections, target):
        """Recursively find sections by heading."""
        for s in sections:
            if s.get("TOCHeading") == target:
                yield s
            for sub in find_sections(s.get("Section", []) or [], target):
                yield sub

    def first_boil_c(sections):
        """Extract first boiling point in Celsius from sections."""
        for sec in find_sections(sections, "Boiling Point"):
            infos = sec.get("Information", []) or []
            for info in infos:
                vals = info.get("Value", {}).get("StringWithMarkup", []) or []
                for v in vals:
                    txt = v.get("String", "")
                    if "°C" in txt:
                        try:
                            num = float(txt.split("°C")[0].strip().replace(",", ""))
                            return num
                        except Exception:
                            pass
                    try:
                        num = float(txt.strip().replace(",", ""))
                        return num
                    except Exception:
                        pass
        return None

    record = data.get("PC_Compounds", [{}])[0]
    props = record.get("props", []) or []

    # Extract formula and molar mass from props
    formula = None
    mw = None
    for p in props:
        urn = p.get("urn", {})
        name = urn.get("name")
        if name == "Molecular Formula" and not formula:
            sval = p.get("value", {}).get("sval")
            if sval:
                formula = sval.strip()
        if name == "Molecular Weight" and mw is None:
            fval = p.get("value", {}).get("fval")
            if fval is not None:
                mw = float(fval)

    # Extract boiling point from experimental properties
    sections = record.get("section", []) or []
    bp_c = first_boil_c(sections)

    return formula, mw, bp_c


class NMVOCFilter(FilterRule):
    """Filter flows by NMVOC classification using boiling point and organic content.
    
    This filter classifies substances as NMVOC (Non-Methane Volatile Organic Compounds)
    based on their boiling point and molecular formula using the chemicals library.
    Can optionally use PubChem online lookups for UNKNOWN flows.
    """
    
    def __init__(
        self,
        nmvoc_status: str = "NMVOC",
        cache_db: str = "nmvoc_cache.sqlite",
        threshold_c: float = DEFAULT_THRESHOLD_C,
        allow_estimates: bool = False,
        online_lookup: bool = False,
    ):
        """
        Args:
            nmvoc_status: Classification to filter for ("NMVOC", "NOT_NMVOC", "UNKNOWN")
            cache_db: Path to SQLite cache database
            threshold_c: Boiling point threshold in Celsius (default 250°C)
            allow_estimates: Allow JOBACK estimates for boiling point
            online_lookup: If True, use PubChem API to enrich UNKNOWN flows
        """
        if not CHEMICALS_AVAILABLE:
            raise ImportError(
                f"chemicals library is required but import failed: {IMPORT_ERROR}\n"
                "Install with: pip install chemicals"
            )
        
        self.nmvoc_status = nmvoc_status
        self.cache_db = cache_db
        self.threshold_c = threshold_c
        self.allow_estimates = allow_estimates
        self.online_lookup = online_lookup
        self._classification_cache = {}
    
    def _make_cache_entry(self, status: str, formula: Optional[str], bp_c: Optional[float], source: str) -> Dict[str, Any]:
        """Helper to create cache dictionary entry."""
        return {
            "status": status,
            "formula": formula,
            "bp_c": bp_c,
            "source": source
        }
    
    def _set_cached(
        self,
        cas: str,
        status: str,
        formula: Optional[str],
        bp_c: Optional[float],
        source: str,
        *,
        write_db: bool = False,
        molar_mass: Optional[float] = None,
    ) -> Tuple[str, Optional[str], Optional[float], Optional[str]]:
        """Update in-memory cache and optionally SQLite; return tuple."""
        self._classification_cache[cas] = self._make_cache_entry(status, formula, bp_c, source)
        if write_db:
            with closing(init_cache(self.cache_db)) as conn:
                cache_put(conn, cas, "", bp_c, formula, molar_mass, source)
        cached = self._classification_cache[cas]
        return cached["status"], cached["formula"], cached["bp_c"], cached["source"]

    def _classify_flow_offline(self, cas: str) -> Tuple[str, Optional[str], Optional[float], Optional[str]]:
        """Classify using offline data only. Returns (status, formula, bp_c, source)."""
        cas = str(cas).strip()

        if cas in self._classification_cache:
            cached = self._classification_cache[cas]
            return cached["status"], cached["formula"], cached["bp_c"], cached["source"]

        if not is_valid_cas(cas):
            return self._set_cached(cas, "UNKNOWN", None, None, "invalid CAS")

        with closing(init_cache(self.cache_db)) as conn:
            cached_db = cache_get(conn, cas)
            if cached_db is not None:
                is_organic = has_carbon(cached_db["formula"])
                status, _ = classify_nmvoc(
                    bp_c=cached_db["bp_c"],
                    has_formula=cached_db["formula"] is not None,
                    is_organic=is_organic,
                    threshold_c=self.threshold_c,
                )
                return self._set_cached(cas, status, cached_db["formula"], cached_db["bp_c"], cached_db["source"])

            bp_c, _ = offline_bp_lookup_c(cas, allow_estimates=self.allow_estimates)
            formula, _ = offline_formula_lookup(cas)
            molar_mass, _ = offline_molar_mass_lookup(cas)

            is_organic = has_carbon(formula)
            status, _ = classify_nmvoc(
                bp_c=bp_c,
                has_formula=formula is not None,
                is_organic=is_organic,
                threshold_c=self.threshold_c,
            )

            source = f"BP:{'chemicals' if bp_c is not None else 'unavailable'} | Formula:{'chemicals' if formula is not None else 'unavailable'}"
            cache_put(conn, cas, "", bp_c, formula, molar_mass, source)

        return self._set_cached(cas, status, formula, bp_c, source)

    def _classify_flow_online(self, cas: str) -> Tuple[str, Optional[str], Optional[float], Optional[str]]:
        """Attempt online classification using PubChem. Returns (status, formula, bp_c, source)."""
        try:
            formula, molar_mass, bp_c = pubchem_fetch(cas, timeout=8.0)
            is_organic = has_carbon(formula)

            status, _ = classify_nmvoc(
                bp_c=bp_c,
                has_formula=formula is not None,
                is_organic=is_organic,
                threshold_c=self.threshold_c,
            )

            source = f"BP:{'pubchem' if bp_c is not None else 'unavailable'} | Formula:{'pubchem' if formula is not None else 'unavailable'}"
            return status, formula, bp_c, source
        except Exception:
            return "UNKNOWN", None, None, "online lookup failed"

    def _classify_flow(self, cas: str) -> str:
        """Classify a single CAS number and return its NMVOC status."""
        cas = str(cas).strip()

        status, formula, bp_c, source = self._classify_flow_offline(cas)

        if status == "UNKNOWN" and self.online_lookup:
            online_status, online_formula, online_bp_c, online_source = self._classify_flow_online(cas)
            if online_status in ("NMVOC", "NOT_NMVOC"):
                status, formula, bp_c, source = self._set_cached(
                    cas,
                    online_status,
                    online_formula,
                    online_bp_c,
                    online_source,
                    write_db=True,
                )

        return status
    
    def apply(self, flow: dict) -> bool:
        """Return True if flow matches the desired NMVOC status."""
        cas = flow.get("CAS")
        if not cas:
            return self.nmvoc_status == "UNKNOWN"
        
        status = self._classify_flow(cas)
        return status == self.nmvoc_status
