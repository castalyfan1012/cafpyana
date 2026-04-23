"""
Constants & utilities for anti-neutrino analysis
"""

from enum import auto, Enum

class TrueType(Enum):
    SIGNAL = auto()
    ANTINU_CC_MULTITRACK = auto()
    ANTINU_CC_OUTOFFV = auto()
    ANTINU_CC_NODECAY = auto()
    ANTINU_NC = auto()
    NU = auto()
    COSMIC = auto()


TYPE_COLORS = {
        TrueType.SIGNAL: "red",
        TrueType.ANTINU_CC_MULTITRACK: "blue",
        TrueType.ANTINU_CC_OUTOFFV: "yellow",
        TrueType.ANTINU_CC_NODECAY: "orange",
        TrueType.ANTINU_NC: "darkgray",
        TrueType.NU: "purple",
        TrueType.COSMIC: "gray",
}
