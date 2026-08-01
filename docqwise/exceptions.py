"""DocQWise exceptions."""


class DocqwiseError(Exception):
    """Base exception for docqwise."""


class ReaderError(DocqwiseError):
    """Error reading a document."""


class UnsupportedFormatError(ReaderError):
    """Document format is not supported."""


class ExtractionError(DocqwiseError):
    """Error extracting content from a document."""


class OCRError(ExtractionError):
    """Error during OCR processing."""


class StoreError(DocqwiseError):
    """Error with vector/database store operations."""


class ConnectionError(DocqwiseError):
    """Error connecting to external service."""


class SchemaError(DocqwiseError):
    """Error with schema detection or validation."""


class PipelineError(DocqwiseError):
    """Error in pipeline execution."""


class CycleDetectedError(PipelineError):
    """Pipeline DAG contains a cycle."""


class ConfigError(DocqwiseError):
    """Invalid configuration."""


class CorrectionError(DocqwiseError):
    """Error applying or storing corrections."""
