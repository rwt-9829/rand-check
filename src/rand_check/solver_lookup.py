"""Baseline probability configuration.

In the original design this module held domain-specific lookup tables.
Now it simply provides a default baseline probability for binary sequence
analysis.  The user can override this via the CLI ``--baseline`` flag or
the ``baseline_prob`` parameter in the Python API.
"""

from __future__ import annotations

# Default baseline probability for a binary outcome.
DEFAULT_BASELINE_PROB: float = 0.5
