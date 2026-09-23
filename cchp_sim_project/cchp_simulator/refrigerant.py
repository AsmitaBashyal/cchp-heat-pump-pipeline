"""
refrigerant.py

Thin wrapper around CoolProp for the two refrigerants used in the CCHP proposal
(Section 2.5): R-410A (predefined CoolProp blend) and R-454B (approximated here
as an HEOS mixture of R-32/R-1234yf at ~68.9%/31.1% by mass, since R-454B is not
yet a predefined pure fluid in open-source CoolProp).

IMPORTANT — flag this honestly in any report: the R-454B mixture model is an
engineering approximation (nominal blend composition, HEOS backend), not a
validated proprietary property correlation. It is adequate for pipeline
development and architecture testing; it should not be cited as a precise
thermophysical reference for R-454B in the thesis itself.
"""

import CoolProp.CoolProp as CP

_R454B_MIX = "HEOS::R32[0.689]&R1234yf[0.311]"

FLUID_MAP = {
    "R410A": "R410A",
    "R454B": _R454B_MIX,
}


def sat_temp_from_pressure(p_pa: float, refrigerant: str) -> float:
    """Saturation temperature (K) at a given pressure (Pa), quality=0 (bubble point)."""
    fluid = FLUID_MAP[refrigerant]
    return CP.PropsSI("T", "P", p_pa, "Q", 0, fluid)


def sat_pressure_from_temp(t_k: float, refrigerant: str) -> float:
    """Saturation pressure (Pa) at a given temperature (K), quality=0 (bubble point)."""
    fluid = FLUID_MAP[refrigerant]
    return CP.PropsSI("P", "T", t_k, "Q", 0, fluid)


def enthalpy_superheated(p_pa: float, t_k: float, refrigerant: str) -> float:
    """Specific enthalpy (J/kg) of superheated vapor at (P, T)."""
    fluid = FLUID_MAP[refrigerant]
    return CP.PropsSI("H", "P", p_pa, "T", t_k, fluid)


def enthalpy_subcooled_liquid(p_pa: float, t_k: float, refrigerant: str) -> float:
    """Specific enthalpy (J/kg) of subcooled liquid at (P, T)."""
    fluid = FLUID_MAP[refrigerant]
    return CP.PropsSI("H", "P", p_pa, "T", t_k, fluid)


def isentropic_outlet_enthalpy(p_in_pa, t_in_k, p_out_pa, refrigerant):
    """Ideal (isentropic) compressor outlet enthalpy, used with an isentropic
    efficiency to get the real outlet enthalpy in cycle_model.py."""
    fluid = FLUID_MAP[refrigerant]
    s_in = CP.PropsSI("S", "P", p_in_pa, "T", t_in_k, fluid)
    return CP.PropsSI("H", "P", p_out_pa, "S", s_in, fluid)


def c_to_k(t_c):
    return t_c + 273.15


def k_to_c(t_k):
    return t_k - 273.15
