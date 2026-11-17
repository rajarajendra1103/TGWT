"""
Property tracking, embedding and gated updates for TGWT.
"""

from .node_properties import NodePropertyTracker
from .edge_properties import EdgePropertyEmbedder
from .gated_updater import GatedNodeUpdater

__all__ = [
    "NodePropertyTracker",
    "EdgePropertyEmbedder",
    "GatedNodeUpdater",
]
