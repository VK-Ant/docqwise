"""Base extractor classes."""
from abc import ABC, abstractmethod
from docqwise.core.document import DocqwiseDocument
from docqwise.core.element import Table, Entity, Relation, FormField
from docqwise.core.field import ExtractionResult

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, document: DocqwiseDocument, **kwargs) -> DocqwiseDocument: ...

class BaseTableExtractor(ABC):
    @abstractmethod
    def extract_tables(self, document: DocqwiseDocument, **kwargs) -> list[Table]: ...

class BaseFormExtractor(ABC):
    @abstractmethod
    def extract_form(self, document: DocqwiseDocument) -> dict[str, FormField]: ...

class BaseEntityExtractor(ABC):
    @abstractmethod
    def extract_entities(self, document: DocqwiseDocument, types: list[str] = None) -> list[Entity]: ...

class BaseRelationExtractor(ABC):
    @abstractmethod
    def extract_relations(self, document: DocqwiseDocument, entities: list[Entity] = None) -> list[Relation]: ...

class BaseFieldExtractor(ABC):
    @abstractmethod
    def extract_fields(self, document: DocqwiseDocument, schema: dict = None) -> ExtractionResult: ...
