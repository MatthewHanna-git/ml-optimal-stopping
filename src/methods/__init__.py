"""Registry of the approximate pricing methods.

Each method exposes price(S0, params) -> Result and fit(), which returns the
trained policy or networks for certification.
"""

from . import dong
from . import dai
from . import chee
from . import dianetti
from . import becker
from . import hure
from . import nn_lsm
from ..reference import lsm

REGISTRY = {
    "dong": dong.price,
    "dai": dai.price,
    "chee": chee.price,
    "dianetti": dianetti.price,
    "becker": becker.price,    # DOS, Becker, Cheridito and Jentzen (2019)
    "hure": hure.price,        # RDBDP, Huré, Pham and Warin (2020)
    "nn_lsm": nn_lsm.price,    # NN-LSM, Lapeyre and Lelong (2021)
    "lsm": lsm.price,          # LSM, Longstaff and Schwartz (2001)
}
