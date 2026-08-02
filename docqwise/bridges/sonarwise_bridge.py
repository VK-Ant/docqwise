"""sonarwise ecosystem bridge."""
class SonarwiseBridge:
    def __init__(self, vellex=None, sonarwise=None):
        self._vellex = vellex
        self._sonarwise = sonarwise

    def ingest_transcripts(self):
        """Ingest audio transcripts from sonarwise as documents."""
        pass
