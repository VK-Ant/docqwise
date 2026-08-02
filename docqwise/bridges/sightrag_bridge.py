"""SightRAG ecosystem bridge."""
class SightRAGBridge:
    def __init__(self, vellex=None, sightrag=None):
        self._vellex = vellex
        self._sightrag = sightrag

    def sync_figures(self):
        """Send extracted figures from documents to SightRAG for visual search."""
        pass  # Implementation depends on SightRAG API
