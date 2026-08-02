"""llmevalkit ecosystem bridge."""
class EvalBridge:
    def __init__(self, vellex=None, evalkit=None):
        self._vellex = vellex
        self._evalkit = evalkit

    def evaluate_extraction(self, documents: str, ground_truth: str, metrics: list = None) -> dict:
        """Evaluate extraction quality using llmevalkit."""
        pass
