"""
NMVOC classification based on boiling point and organic content.
Uses ONLY offline chemicals library - no external API calls.
"""

import datetime as dt
import sqlite3
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


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


# Import chemicals library (required)
try:
    from chemicals.phase_change import Tb, Tb_methods
    from chemicals.identifiers import MW
    from chemicals import search_chemical  # Use search_chemical instead
    CHEMICALS_AVAILABLE = True
except ImportError as e:
    CHEMICALS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# SQLite cache helpers

def init_cache(db_path: str) -> sqlite3.Connection:
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
    conn.execute(
        "INSERT OR REPLACE INTO results (cas, name, bp_c, formula, molar_mass_g_mol, source, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (cas, name, bp_c, formula, molar_mass_g_mol, source, dt.datetime.now(dt.timezone.utc).isoformat()),
    )
    conn.commit()


# Offline lookups only

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
        # Use search_chemical to get chemical info
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


# Organic and carbon detection

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
    # Match C not followed by lowercase letter (to exclude Cl, Ca, Cd, etc.)
    match = re.search(r'C(?![a-z])', formula)
    return match is not None


# Classification logic

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
    
    Args:
        bp_c: Boiling point in Celsius (or None if unavailable)
        has_formula: Whether formula is available
        is_organic: Whether formula contains carbon
        threshold_c: Boiling point threshold
    
    Returns:
        Tuple of (status, reason)
    """
    # Case 1: BP >= 250°C -> NOT_NMVOC (regardless of formula/organic)
    if bp_c is not None and bp_c >= threshold_c:
        return "NOT_NMVOC", f"BP {bp_c:.2f}°C >= {threshold_c}°C"
    
    # Case 2: Formula available and not organic -> NOT_NMVOC
    if has_formula and not is_organic:
        return "NOT_NMVOC", "Not organic (no carbon in formula)"
    
    # Case 3: Formula available AND organic AND BP < 250°C -> NMVOC
    if has_formula and is_organic and bp_c is not None and bp_c < threshold_c:
        return "NMVOC", f"Organic + BP {bp_c:.2f}°C < {threshold_c}°C"
    
    # Case 4: No formula -> UNKNOWN (unless BP >= 250°C, handled in Case 1)
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


# --- CAS validation helpers --------------------------------------------------

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
    if not cas:
        return False
    cas = cas.strip()
    if not CAS_REGEX.match(cas):
        return False
    return cas_checksum_ok(cas)

# Main DataFrame classifier

def classify_nmvoc_bp(
    df: pd.DataFrame,
    cache_db: str = "nmvoc_cache.sqlite",
    cas_col: str = "CAS",
    name_col: str = "name",
    formula_col: Optional[str] = None,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    allow_estimates: bool = False,
    progress_every: int = 50,
) -> pd.DataFrame:
    """
    Pass A (fast, deterministic):
    - Validate CAS
    - Use cache + offline chemicals lookups
    - Never call online sources
    """
    if not CHEMICALS_AVAILABLE:
        raise ImportError(
            f"chemicals library is required but import failed: {IMPORT_ERROR}\n"
            "Install with: pip install chemicals\n"
            "Or in notebook: !pip install chemicals"
        )
    if cas_col not in df.columns or name_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{cas_col}' and '{name_col}'")

    out = df.copy()
    out[cas_col] = out[cas_col].astype(str).str.strip()
    out[name_col] = out[name_col].astype(str).str.strip()

    out["bp_c"] = pd.NA
    out["formula"] = None
    out["is_organic"] = False
    out["nmvoc_status"] = "UNKNOWN"
    out["reason"] = ""
    out["molar_mass"] = pd.NA
    out["source"] = ""

    conn = init_cache(cache_db)
    unique_cas = out[cas_col].dropna().astype(str).str.strip().unique().tolist()
    cas_results: Dict[str, Dict[str, Any]] = {}
    processed = 0

    for cas in unique_cas:
        # Early validation
        if not is_valid_cas(cas):
            cas_results[cas] = {
                "cas": cas,
                "bp_c": None,
                "formula": None,
                "is_organic": False,
                "nmvoc_status": "UNKNOWN",
                "reason": "invalid CAS",
                "molar_mass": None,
                "source": "BP:[skipped invalid] | Formula:[skipped invalid]",
            }
            continue

        # Cache hit
        cached = cache_get(conn, cas)
        if cached is not None:
            is_organic = has_carbon(cached["formula"])
            status, reason = classify_nmvoc(
                bp_c=cached["bp_c"],
                has_formula=cached["formula"] is not None,
                is_organic=is_organic,
                threshold_c=threshold_c,
            )
            cas_results[cas] = {
                "cas": cas,
                "bp_c": cached["bp_c"],
                "formula": cached["formula"],
                "is_organic": is_organic,
                "nmvoc_status": status,
                "reason": reason,
                "molar_mass": cached["molar_mass"],
                "source": cached["source"],
            }
            continue

        # Offline lookups only (Pass A)
        bp_c, bp_source = offline_bp_lookup_c(cas, allow_estimates=allow_estimates)
        formula, formula_source = offline_formula_lookup(cas)
        molar_mass, _ = offline_molar_mass_lookup(cas)

        is_organic = has_carbon(formula)

        status, reason = classify_nmvoc(
            bp_c=bp_c,
            has_formula=formula is not None,
            is_organic=is_organic,
            threshold_c=threshold_c,
        )

        source_parts = []
        source_parts.append(f"BP:{bp_source if bp_c is not None else '[' + bp_source + ']'}")
        source_parts.append(f"Formula:{formula_source if formula is not None else '[' + formula_source + ']'}")
        source = " | ".join(source_parts)

        cas_results[cas] = {
            "cas": cas,
            "bp_c": bp_c,
            "formula": formula,
            "is_organic": is_organic,
            "nmvoc_status": status,
            "reason": reason,
            "molar_mass": molar_mass,
            "source": source,
        }

        cache_put(conn, cas, "", bp_c, formula, molar_mass, source)

        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"Processed {processed}/{len(unique_cas)} unique CAS (offline only)...")

    conn.close()

    # Map results back
    out["bp_c"] = out[cas_col].map(lambda x: cas_results.get(str(x).strip(), {}).get("bp_c"))
    out["formula"] = out[cas_col].map(lambda x: cas_results.get(str(x).strip(), {}).get("formula"))
    out["is_organic"] = out[cas_col].map(lambda x: cas_results.get(str(x).strip(), {}).get("is_organic", False))
    out["nmvoc_status"] = out[cas_col].map(lambda x: cas_results.get(str(x).strip(), {}).get("nmvoc_status", "UNKNOWN"))
    out["reason"] = out[cas_col].map(lambda x: cas_results.get(str(x).strip(), {}).get("reason", ""))
    out["molar_mass"] = out[cas_col].map(lambda x: cas_results.get(str(x).strip(), {}).get("molar_mass"))
    out["source"] = out[cas_col].map(lambda x: cas_results.get(str(x).strip(), {}).get("source", ""))

    return out

# Placeholder for Pass B (optional, online enrichment), to be implemented separately.
# def enrich_unknowns_online(df_unknown: pd.DataFrame, ...):
#     ...
