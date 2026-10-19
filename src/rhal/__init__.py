"""RobustHFT-AgentLab (rhal).

A defensive, offline simulation framework for testing and improving LLM-based
trading agents under subtle financial misinformation.

SAFETY BOUNDARY
---------------
* All misinformation is synthetic, generated and consumed inside the simulator,
  and every record carries ``provenance="synthetic"``.
* There is no broker connection, no order routing, and no live data path.
* Nothing in this package constitutes investment advice.
"""

from __future__ import annotations

__version__ = "0.1.0"
