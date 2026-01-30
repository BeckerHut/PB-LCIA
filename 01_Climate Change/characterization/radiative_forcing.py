import numpy as np
import pandas as pd

from classes import CharacterizedRow

# ============================================================================
# Constants
# ============================================================================
M_AIR = 28.97  # g/mol, dry air
M_ATMOSPHERE = 5.135e18  # kg [Trenberth and Smith, 2005]
M_CO2 = 44.01  # g/mol
RAD_EFF_CO2_PPB = 1.33e-5  # W/m2/ppb; IPCC AR6 Table 7.15


# ============================================================================
# Helper Functions
# ============================================================================
def _validate_parameters(**kwargs) -> None:
    """
    Validate that required parameters are not None.
    """
    for param_name, param_value in kwargs.items():
        if param_value is None:
            formatted_name = param_name.replace('_', ' ').title()
            raise ValueError(f"{formatted_name} not available. Unable to compute radiative forcing.")


def _convert_rad_eff_ppb_to_kg(rad_eff_ppb: float, mol_mass: float) -> float:
    """
    Convert radiative efficiency from ppb to kg units.
    
    Parameters
    ----------
    rad_eff_ppb : float
        Radiative efficiency in W/m2/ppb
    mol_mass : float
        Molar mass in g/mol
        
    Returns
    -------
    float
        Radiative efficiency in W/m2/kg
    """
    return rad_eff_ppb * M_AIR / mol_mass * 1e9 / M_ATMOSPHERE


def _create_time_axis(series, period: int) -> np.ndarray:
    """
    Create time axis for characterization.
    
    Parameters
    ----------
    series : pandas.Series
        Input series with date field
    period : int
        Number of years
        
    Returns
    -------
    np.ndarray
        Array of dates from series.date to series.date + period years
    """
    date_beginning: np.datetime64 = series.date.to_numpy()
    return date_beginning + np.arange(
        start=0, stop=period, dtype="timedelta64[Y]"
    ).astype("timedelta64[s]")


def _create_characterized_row(
    date: np.ndarray, amount: np.ndarray, series
) -> CharacterizedRow:
    """
    Create CharacterizedRow with standard fields.
    
    Parameters
    ----------
    date : np.ndarray
        Array of datetime64[s]
    amount : np.ndarray
        Array of forcing values
    series : pandas.Series
        Input series with flow and activity fields
        
    Returns
    -------
    CharacterizedRow
        Characterized row object
    """
    return CharacterizedRow(
        date=date,
        amount=amount,
        flow=series.flow,
        activity=series.activity,
    )


def IRF_co2(year) -> float:
    """
    Integrated Impulse Response Function (IRF) of CO2

    Parameters
    ----------
    year : int
        The year after emission for which the integrated IRF is calculated.

    Returns
    -------
    float
        The integrated IRF value for the given year.

    """
    alpha_0, alpha_1, alpha_2, alpha_3 = 0.2173, 0.2240, 0.2824, 0.2763
    tau_1, tau_2, tau_3 = 394.4, 36.54, 4.304
    exponentials = lambda year, alpha, tau: alpha * tau * (1 - np.exp(-year / tau))
    return (
        alpha_0 * year
        + exponentials(year, alpha_1, tau_1)
        + exponentials(year, alpha_2, tau_2)
        + exponentials(year, alpha_3, tau_3)
    )


def characterize_co2(
    series,
    period,
) -> CharacterizedRow:
    """
    Calculate (marginal) radiative forcing (RF) from CO2 by computing the difference of two consecutive cumulative AGWP values for each year in a given period.
    Units are watts/square meter/kilogram of CO2.

    Based on characterize_co2 from brightway-lca/dynamic_characterization

    Parameters
    ----------
     series 
        Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).
    
     period : int
        Time period for calculation (number of years)
    

    
    Returns
    -------
    CharacterizedRow 
    """

    # Validate required parameters
    _validate_parameters(period=period)
    
    # Convert CO2 radiative efficiency from ppb to kg
    rad_eff_kg = _convert_rad_eff_ppb_to_kg(RAD_EFF_CO2_PPB, M_CO2)

    # Create time axis
    dates_characterized: np.ndarray = _create_time_axis(series, period)

    # Calculate decay multipliers for each year
    decay_multipliers: np.ndarray = np.array(
        [rad_eff_kg * IRF_co2(year) for year in range(period)]
    )

    # Calculate cumulative AGWP
    forcing = np.array(series.amount * decay_multipliers, dtype="float64")

    # Convert cumulative AGWP to marginal RF per year
    forcing = np.diff(forcing, prepend=0)

    return _create_characterized_row(
        date=np.array(dates_characterized, dtype="datetime64[s]"),
        amount=forcing,
        series=series,
    )


def characterize_voc_nolt(
    series,
    period,
    mol_mass,
    carbon_count,
) -> CharacterizedRow:
    """
    Calculate (marginal) radiative forcing (RF) from a VOC where no lifetime is available. 
    It is assumed that the VOC is instantaneously and comlpletely oxidized to CO2.
    
    This function applies the CO2 characterization and scales it by the fractional release factor.
    
    Units are watts/square meter/kilogram of VOC.

    Parameters
    ----------
    series 
        Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).
    
    period : int
        Time period for calculation (number of years)

    mol_mass : float
        Molar mass of VOC (g/mol)
    
    carbon_count : int
        Number of carbon atoms per molecule of VOC

    Returns
    -------
    CharacterizedRow 
    """
    
    # Validate required parameters
    _validate_parameters(period=period, mol_mass=mol_mass, carbon_count=carbon_count)
    
    # Calculate fractional release factor (kg-CO2 / kg-VOC)
    fr = carbon_count * (M_CO2 / mol_mass)
    
    # Get CO2 characterization result
    co2_result = characterize_co2(
        series=series,
        period=period,
    )
    
    # Scale the forcing by fractional release factor
    scaled_forcing = co2_result.amount * fr
    
    return _create_characterized_row(
        date=co2_result.date,
        amount=scaled_forcing,
        series=series,
    )


def characterize_other_ghg(
    series,
    period,
    rad_eff_ppb,
    tau,
    mol_mass,
) -> CharacterizedRow:
    """
    Calculate the (marginal) radiative forcing from a GHG for each year in a given period.

    Parameters
    ----------
    series 
        Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).

    period : int
        Time period for calculation (number of years)
           
    rad_eff_ppb : float
        Radiative efficiency for GHG (W/m2/ppb)
    
    tau : float
        Lifetime of GHG (years)
           
    mol_mass : float
        Molar mass of GHG (g/mol)


    Returns
    -------
    CharacterizedRow 
    """

    # Validate required parameters
    _validate_parameters(
        period=period, rad_eff_ppb=rad_eff_ppb, mol_mass=mol_mass, tau=tau
    )
    
    # Convert radiative efficiency from ppb to kg
    radiative_efficiency_kg = _convert_rad_eff_ppb_to_kg(rad_eff_ppb, mol_mass)

    # Create time axis
    dates_characterized: np.ndarray = _create_time_axis(series, period)

    # Calculate decay multipliers for each year
    decay_multipliers: np.ndarray = np.array(
        [
            radiative_efficiency_kg * tau * (1 - np.exp(-year / tau))
            for year in range(period)
        ]
    )

    # Calculate cumulative AGWP
    forcing = np.array(series.amount * decay_multipliers, dtype="float64")

    # Convert cumulative AGWP to marginal RF per year
    forcing = np.diff(forcing, prepend=0)

    return _create_characterized_row(
        date=np.array(dates_characterized, dtype="datetime64[s]"),
        amount=forcing,
        series=series,
    )


def characterize_voc_lt(
    series,
    period,
    rad_eff_ppb,
    tau,
    mol_mass,
    carbon_count,
) -> CharacterizedRow:
    """
    Calculate the total (marginal) radiative forcing of VOC combining both direct and indirect effects.
    
    This function calculates:
    1. Direct RF: Direct radiative forcing from the VOC itself
    2. Indirect RF: Indirect radiative forcing from CO2 formed by VOC oxidation
    
    The results are combined by adding the forcing values at each time step.

    Parameters
    ----------
    series 
        Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).

    period : int
        Time period for calculation (number of years)
           
    rad_eff_ppb : float
        Radiative efficiency for VOC (W/m2/ppb)
    
    tau : float
        Lifetime of VOC (years)
           
    mol_mass : float
        Molar mass of VOC (g/mol)

    carbon_count : int
        Number of carbon atoms per molecule of VOC

    Returns
    -------
    CharacterizedRow 
    """

    # Calculate direct impact
    direct_result = voc_lt_direct_impact(
        series=series,
        period=period,
        rad_eff_ppb=rad_eff_ppb,
        tau=tau,
        mol_mass=mol_mass,
    )
    
    # Calculate indirect impact
    indirect_result = voc_lt_indirect_impact(
        series=series,
        period=period,
        tau=tau,
        mol_mass=mol_mass,
        carbon_count=carbon_count,
    )
    
    # Combine the forcing values (element-wise addition)
    combined_forcing = direct_result.amount + indirect_result.amount
    
    # Return CharacterizedRow with combined forcing
    return CharacterizedRow(
        date=direct_result.date,
        amount=combined_forcing,
        flow=series.flow,
        activity=series.activity,
    )



def voc_lt_direct_impact(
    series,
    period,
    rad_eff_ppb,
    tau,
    mol_mass,
) -> CharacterizedRow:
    """
    Calculate the (marginal) radiative forcing of direct effects from the VOC for each year in a given period.

    Parameters
    ----------
    series 
        Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).

    period : int
        Time period for calculation (number of years)
           
    rad_eff_ppb : float
        Radiative efficiency for VOC (W/m2/ppb)
    
    tau : float
        Lifetime of VOC (years)
           
    mol_mass : float
        Molar mass of VOC (g/mol)


    Returns
    -------
    CharacterizedRow 
    """

    # Validate required parameters
    _validate_parameters(
        period=period, rad_eff_ppb=rad_eff_ppb, mol_mass=mol_mass, tau=tau
    )
    
    # Convert radiative efficiency from ppb to kg
    radiative_efficiency_kg = _convert_rad_eff_ppb_to_kg(rad_eff_ppb, mol_mass)

    # Create time axis
    dates_characterized: np.ndarray = _create_time_axis(series, period)

    # Calculate decay multipliers for each year
    decay_multipliers: np.ndarray = np.array(
        [
            radiative_efficiency_kg * tau * (1 - np.exp(-year / tau))
            for year in range(period)
        ]
    )

    # Calculate cumulative AGWP
    forcing = np.array(series.amount * decay_multipliers, dtype="float64")

    # Convert cumulative AGWP to marginal RF per year
    forcing = np.diff(forcing, prepend=0)

    return _create_characterized_row(
        date=np.array(dates_characterized, dtype="datetime64[s]"),
        amount=forcing,
        series=series,
    )


def voc_lt_indirect_impact(
    series,
    period,
    tau,
    mol_mass,
    carbon_count,
) -> CharacterizedRow:
    """
    Calculate marginal radiative forcing (RF) of indirect effects from the CO2 formed by a VOC pulse emission,
    using discrete CO2 creation (a series of small CO2 pulses).

    Parameters
    ----------
    series 
        Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).

    period : int
        Time period for calculation (number of years)
           
    tau : float
        Lifetime of VOC (years)
           
    mol_mass : float
        Molar mass of VOC (g/mol)

    carbon_count : int
        Number of carbon atoms per molecule of VOC


    Returns
    -------
    CharacterizedRow 
    """
    
    # Validate required parameters
    _validate_parameters(
        period=period, mol_mass=mol_mass, tau=tau, carbon_count=carbon_count
    )
    
    # Convert CO2 radiative efficiency from ppb to kg
    rad_eff_co2_kg = _convert_rad_eff_ppb_to_kg(RAD_EFF_CO2_PPB, M_CO2)

    # Calculate fractional release factor (kg-CO2 / kg-VOC)
    fr = carbon_count * (M_CO2 / mol_mass)
    
    # Create time axis
    dates_characterized: np.ndarray = _create_time_axis(series, period)

    # Calculation of CO2 AGWP kernel for a unit CO2 pulse emitted at time 0
    kernel_agwp_per_kgco2: np.ndarray = np.array(
        [rad_eff_co2_kg * IRF_co2(year) for year in range(period)],
        dtype="float64",
    )

    # Discrete CO2 formation from VOC decay
    # CO2 formed during year k: ΔCO2_k = E * f_co2 * (exp(-k/tau) - exp(-(k+1)/tau))
    years = np.arange(period, dtype="float64")
    decay_start = np.exp(-years / tau)
    decay_end = np.exp(-(years + 1.0) / tau)
    delta_co2 = series.amount * fr * (decay_start - decay_end)  # kg-CO2 formed in each year-bin

    # Convolution: cumulative AGWP at each year n from all past formed CO2 pulses
    # cumulative_agwp[n] = sum_k delta_co2[k] * kernel_agwp_per_kgco2[n-k]
    cumulative_agwp = np.convolve(delta_co2, kernel_agwp_per_kgco2)[:period].astype("float64")

    # Convert cumulative AGWP to marginal RF per year
    forcing = np.diff(cumulative_agwp, prepend=0)

    return _create_characterized_row(
        date=np.array(dates_characterized, dtype="datetime64[s]"),
        amount=forcing,
        series=series,
    )