"""
Constants & utilities for anti-neutrino analysis
"""

from enum import auto, Enum

class TrueType(Enum):
    SIGNAL = auto()
    ANTINU_CC_NODECAY = auto()
    NU_CC = auto()
    NC = auto()
    NU_OUT_OF_FV = auto()
    COSMIC = auto()


TYPE_COLORS = {
        TrueType.SIGNAL: "red",
        TrueType.ANTINU_CC_NODECAY: "orange",
        TrueType.NU_CC: "purple",
        TrueType.NC: "darkgray",
        TrueType.NU_OUT_OF_FV: "yellow",
        TrueType.COSMIC: "gray",
}

TYPE_LABELS = {
        TrueType.SIGNAL: r"$\bar{\nu}_{\mu}$ CC signal",
        TrueType.ANTINU_CC_NODECAY: r"$\bar{\nu}_{\mu}$, no decay in FV",
        TrueType.NU_CC: r"Other $\nu$ CC",
        TrueType.NC: "NC",
        TrueType.NU_OUT_OF_FV: "OOFV",
        TrueType.COSMIC: "Cosmic"
}
