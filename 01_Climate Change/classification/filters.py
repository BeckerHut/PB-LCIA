import pandas as pd # for dataframe operations
import re # for regex matching pattern 
import sqlite3 # for local database caching
import datetime as dt # for adding timestemps
import requests # for http requests
from abc import ABC, abstractmethod # for creating an abstract base class
from typing import Optional, Tuple, Dict, Any # for improving code readability
from contextlib import closing # for context-managed SQLite connections

# Import chemicals library (required for VOC classification)
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
    """Filter flows by multiple name substrings (case-insensitive, OR logic)"""
    
    def __init__(self, substrings: list):
        self.substrings = [s.lower() for s in substrings]
    
    def apply(self, flow: dict) -> bool:
        name = flow.get("name", "").lower()
        return any(substring in name for substring in self.substrings)
    # checks if ANY substring is in the name    







def apply_filter(df: pd.DataFrame, filter_rule: FilterRule) -> pd.DataFrame:
    """Apply a single filter rule to the dataframe and return filtered results"""
    # Convert dataframe rows to dict format for filtering
    filtered_rows = []
    extra_cols = {}  # Store extra columns like carbon_atoms
    
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
            # Capture any extra columns added by the filter
            for key in flow_dict:
                if key not in ["name", "categories", "CAS"]:
                    if key not in extra_cols:
                        extra_cols[key] = {}
                    extra_cols[key][idx] = flow_dict[key]
    
    df_filtered = df.loc[filtered_rows].reset_index(drop=True)
    
    # Add extra columns to the filtered dataframe
    for col_name, col_data in extra_cols.items():
        # Map the indices back (filtered_rows contains original indices)
        df_filtered[col_name] = [col_data.get(orig_idx) for orig_idx in filtered_rows]
    
    return df_filtered


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
# VOC Classification Filter
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
    """Initialize SQLite cache for VOC classification results."""
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
    """Retrieve cached VOC classification result."""
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
    """Store VOC classification result in cache."""
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


def count_carbon_atoms(formula: Optional[str]) -> int:
    """Count the number of carbon atoms in a molecular formula.
    
    Args:
        formula: Molecular formula string (e.g., "C6H14", "NaCl", "H2O")
    
    Returns:
        Number of carbon atoms (0 if no carbon or formula is None)
    """
    if not formula:
        return 0
    
    formula = str(formula).strip()
    # Find all occurrences of C followed by optional digits, but not Cl, Ca, Cd, etc.
    matches = re.findall(r'C(?![a-z])(\d*)', formula)
    
    total_carbons = 0
    for match in matches:
        if match == '':
            total_carbons += 1  # Single C without a number means 1
        else:
            total_carbons += int(match)
    
    return total_carbons


def classify_voc(
    bp_c: Optional[float],
    has_formula: bool,
    is_organic: bool,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    flow_name: Optional[str] = None,
) -> Tuple[str, str]:
    """Classify compound as VOC, NOT_VOC, or UNKNOWN.
    
    Logic:
    - VOC: (organic AND bp < 250°C), 
            also includes carbon monoxide
    - NOT_VOC: bp >= 250°C OR (formula exists AND not organic) 
            and includes all substances with carbon content but considered not organic
    - UNKNOWN: formula unavailable AND (bp unavailable OR bp < 250°C)
    """
    # Inclusion list for Step 1
    """CO is not a VOC but acts also as CO2 precursor 
    and is therefore characterized the same way"""

    INCLUDED_NAMES = {
        "carbon monoxide",
    }

    # Exclusion list for Step 2
    """These substances contain carbon but are not considered VOCs"""

    EXCLUDED_NAMES = {
        "Ammonium carbonate",
        "Bicarbonate",
        "Boron carbide",
        "Carbon-14",
        "Carbon disulfide",
        "Carbonate",
        "Carbonyl sulfide",
        "Cyanide",
        "Elemental carbon",
        "Graphite",
        "Lithium carbonate",
        "Phosgene",
        "Thiocyanate"
    }
    
    # Case 1: Check if flow name matches inclusion list -> VOC
    if flow_name is not None:
        flow_name_lower = flow_name.lower()
        if any(name in flow_name_lower for name in INCLUDED_NAMES):
            return "VOC", f"Included by name: {flow_name}"

    # Case 2: Check if flow name matches exclusion list -> NOT_VOC
    if flow_name is not None and flow_name in EXCLUDED_NAMES:
        return "NOT_VOC", f"Excluded by name: {flow_name}"
    
    # Case 3: BP >= 250°C -> NOT_VOC
    if bp_c is not None and bp_c >= threshold_c:
        return "NOT_VOC", f"BP {bp_c:.2f}°C >= {threshold_c}°C"
    
    # Case 4: Formula available and not organic -> NOT_VOC
    if has_formula and not is_organic:
        return "NOT_VOC", "Not organic (no carbon in formula)"
    
    # Case 5: Formula available AND organic AND BP < 250°C -> VOC
    if has_formula and is_organic and bp_c is not None and bp_c < threshold_c:
        return "VOC", f"Organic + BP {bp_c:.2f}°C < {threshold_c}°C"
    
    # Case 6: No formula -> UNKNOWN
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
    """Fetch formula, molar mass, and boiling point from PubChem.

    Notes:
        PubChem's "compound" endpoints (PUG REST) reliably provide formula/MW via the
        Property API, but experimental boiling points are typically exposed via the
        PUG View (record) API. This function therefore:

        1) Resolves CAS (RN) -> CID
        2) Fetches MolecularFormula + MolecularWeight via the Property API
        3) Tries to extract a Boiling Point (°C) from the PUG View record

    Args:
        cas: CAS number string (ideally normalized, e.g. '7732-18-5')
        timeout: Request timeout in seconds

    Returns:
        Tuple of (formula, molar_mass_g_mol, bp_c) where any may be None.

    Raises:
        RuntimeError: if the CAS cannot be resolved or PubChem is unavailable.
    """
    cas = str(cas).strip()

    # 1) Resolve CAS RN -> CID
    url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/rn/{cas}/cids/JSON"
    try:
        r = requests.get(url_cid, timeout=timeout)
        r.raise_for_status()
        cid_data = r.json()
    except Exception as e:
        raise RuntimeError(f"PubChem RN->CID lookup failed ({type(e).__name__})") from e

    cids = (cid_data.get("IdentifierList", {}) or {}).get("CID", []) or []
    if not cids:
        raise RuntimeError("PubChem returned no CID for this CAS (RN)")

    cid = cids[0]

    # 2) Fetch formula + MW (reliable)
    formula: Optional[str] = None
    mw: Optional[float] = None
    url_props = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight/JSON"
    try:
        r = requests.get(url_props, timeout=timeout)
        r.raise_for_status()
        props = r.json().get("PropertyTable", {}).get("Properties", []) or []
        if props:
            p0 = props[0] or {}
            formula = p0.get("MolecularFormula")
            mw_val = p0.get("MolecularWeight")
            if mw_val is not None:
                try:
                    mw = float(mw_val)
                except Exception:
                    mw = None
    except Exception:
        # Keep going; boiling point might still be available
        pass

    # 3) Extract boiling point from PUG View record (best-effort)
    bp_c: Optional[float] = None
    url_view = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    try:
        r = requests.get(url_view, timeout=timeout)
        r.raise_for_status()
        view = r.json()
    except Exception:
        return formula, mw, None

    def iter_sections(sec_list):
        for s in sec_list or []:
            yield s
            yield from iter_sections(s.get("Section"))

    record = view.get("Record", {}) or {}
    for sec in iter_sections(record.get("Section")):
        if sec.get("TOCHeading") != "Boiling Point":
            continue
        infos = sec.get("Information", []) or []
        for info in infos:
            vals = (info.get("Value", {}) or {}).get("StringWithMarkup", []) or []
            for v in vals:
                txt = (v.get("String") or "").strip()
                if not txt:
                    continue
                # Common patterns: "117 °C", "117.3 °C", sometimes with commas
                if "°C" in txt:
                    left = txt.split("°C")[0].strip()
                    left = left.replace(",", "")
                    try:
                        bp_c = float(left)
                        break
                    except Exception:
                        pass
            if bp_c is not None:
                break
        if bp_c is not None:
            break

    return formula, mw, bp_c



class VOCFilter(FilterRule):
    """Filter flows by VOC classification using boiling point and organic content.
    
    This filter classifies substances as VOC (Volatile Organic Compounds)
    based on their boiling point and molecular formula using the chemicals library.
    Can optionally use PubChem online lookups for UNKNOWN flows.
    """
    
    def __init__(
        self,
        voc_status: str = "VOC",
        cache_db: str = "voc_cache.sqlite",
        threshold_c: float = DEFAULT_THRESHOLD_C,
        allow_estimates: bool = False,
        online_lookup: bool = False,
    ):
        """
        Args:
            voc_status: Classification to filter for ("VOC", "NOT_VOC", "UNKNOWN")
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
        
        self.voc_status = voc_status
        self.cache_db = cache_db
        self.threshold_c = threshold_c
        self.allow_estimates = allow_estimates
        self.online_lookup = online_lookup
        self._classification_cache = {}
    
    def _make_cache_entry(self, status: str, formula: Optional[str], bp_c: Optional[float], source: str, molar_mass: Optional[float] = None) -> Dict[str, Any]:
        """Helper to create cache dictionary entry."""
        return {
            "status": status,
            "formula": formula,
            "bp_c": bp_c,
            "source": source,
            "molar_mass": molar_mass
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
        self._classification_cache[cas] = self._make_cache_entry(status, formula, bp_c, source, molar_mass)
        if write_db:
            with closing(init_cache(self.cache_db)) as conn:
                cache_put(conn, cas, "", bp_c, formula, molar_mass, source)
        cached = self._classification_cache[cas]
        return cached["status"], cached["formula"], cached["bp_c"], cached["source"]

    def _classify_flow_offline(self, cas: str, flow_name: Optional[str] = None) -> Tuple[str, Optional[str], Optional[float], Optional[str]]:
        """Classify using offline data only. Returns (status, formula, bp_c, source)."""
        cas = normalize_cas(cas) if cas is not None else None
        cas = str(cas).strip() if cas is not None else ""

        if cas in self._classification_cache:
            cached = self._classification_cache[cas]
            return cached["status"], cached["formula"], cached["bp_c"], cached["source"]

        if not is_valid_cas(cas):
            return self._set_cached(cas, "UNKNOWN", None, None, "invalid CAS")

        with closing(init_cache(self.cache_db)) as conn:
            cached_db = cache_get(conn, cas)
            if cached_db is not None:
                is_organic = has_carbon(cached_db["formula"])
                status, _ = classify_voc(
                    bp_c=cached_db["bp_c"],
                    has_formula=cached_db["formula"] is not None,
                    is_organic=is_organic,
                    threshold_c=self.threshold_c,
                    flow_name=flow_name,
                )
                return self._set_cached(cas, status, cached_db["formula"], cached_db["bp_c"], cached_db["source"], molar_mass=cached_db.get("molar_mass"))

            bp_c, _ = offline_bp_lookup_c(cas, allow_estimates=self.allow_estimates)
            formula, _ = offline_formula_lookup(cas)
            molar_mass, _ = offline_molar_mass_lookup(cas)

            is_organic = has_carbon(formula)
            status, _ = classify_voc(
                bp_c=bp_c,
                has_formula=formula is not None,
                is_organic=is_organic,
                threshold_c=self.threshold_c,
                flow_name=flow_name,
            )

            source = f"BP:{'chemicals' if bp_c is not None else 'unavailable'} | Formula:{'chemicals' if formula is not None else 'unavailable'}"
            cache_put(conn, cas, "", bp_c, formula, molar_mass, source)

        return self._set_cached(cas, status, formula, bp_c, source, molar_mass=molar_mass)

    def _classify_flow_online(self, cas: str, flow_name: Optional[str] = None) -> Tuple[str, Optional[str], Optional[float], Optional[str], Optional[float]]:
        """Attempt online classification using PubChem. Returns (status, formula, bp_c, source, molar_mass)."""
        cas = normalize_cas(cas) if cas is not None else None
        cas = str(cas).strip() if cas is not None else ""
        try:
            formula, molar_mass, bp_c = pubchem_fetch(cas, timeout=8.0)
            is_organic = has_carbon(formula)

            status, _ = classify_voc(
                bp_c=bp_c,
                has_formula=formula is not None,
                is_organic=is_organic,
                threshold_c=self.threshold_c,
                flow_name=flow_name,
            )

            source = f"BP:{'pubchem' if bp_c is not None else 'unavailable'} | Formula:{'pubchem' if formula is not None else 'unavailable'}"
            return status, formula, bp_c, source, molar_mass
        except Exception as e:
            return "UNKNOWN", None, None, f"online lookup failed: {type(e).__name__}: {e}", None

    def _classify_flow(self, cas: str, flow_name: Optional[str] = None) -> str:
        """Classify a single CAS number and return its VOC status."""
        cas = normalize_cas(cas) if cas is not None else None
        cas = str(cas).strip() if cas is not None else ""

        status, formula, bp_c, source = self._classify_flow_offline(cas, flow_name=flow_name)

        if status == "UNKNOWN" and self.online_lookup:
            online_status, online_formula, online_bp_c, online_source, online_molar_mass = self._classify_flow_online(cas, flow_name=flow_name)

            # If the online lookup returns *any* enrichment (formula and/or BP), cache it
            # even if the final classification remains UNKNOWN. This makes the behavior
            # observable and avoids repeating lookups in future runs.
            if (
                online_status in ("VOC", "NOT_VOC")
                or online_formula is not None
                or online_bp_c is not None
            ):
                status, formula, bp_c, source = self._set_cached(
                    cas,
                    online_status,
                    online_formula,
                    online_bp_c,
                    online_source,
                    write_db=True,
                    molar_mass=online_molar_mass,
                )

        return status
    
    def apply(self, flow: dict) -> bool:
        """Return True if flow matches the desired VOC status and add carbon atom count."""
        cas = normalize_cas(flow.get("CAS"))
        if not cas:
            return self.voc_status == "UNKNOWN"
        
        flow_name = flow.get("name")
        status = self._classify_flow(cas, flow_name=flow_name)
        
        # Add carbon atom count to the flow
        formula = self._classification_cache.get(cas, {}).get("formula")
        flow["carbon_atoms"] = count_carbon_atoms(formula)
        
        return status == self.voc_status
