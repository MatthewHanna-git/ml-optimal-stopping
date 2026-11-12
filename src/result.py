"""The Result object returned by every pricing method.

Every method returns a Result rather than a bare float, so experiment scripts
can treat all methods alike. `bounds` holds a (lower, upper) interval where a
method reports one; `diagnostics` holds anything method-specific.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


@dataclass
class Result:
    value: float                                   # the price estimate
    runtime: float                                 # wall-clock seconds (GPU-synchronised where relevant)
    bounds: Optional[Tuple[float, float]] = None   # (lower, upper), where reported
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        s = f"value={self.value:.4f}  runtime={self.runtime:.2f}s"
        if self.bounds is not None:
            s += f"  bounds=({self.bounds[0]:.4f}, {self.bounds[1]:.4f})"
        return s
