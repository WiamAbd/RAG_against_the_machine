import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A chunk of data."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question, generating an ID when none is supplied."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a reference question with its answer and supporting sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Contain a collection of answered or unanswered questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store ranked source locations retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store a generated answer alongside its question and retrieved sources."""

    answer: str


class StudentSearchResults(BaseModel):
    """Contain batch retrieval results and the requested sources per question."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Contain batch retrieval results with generated answers and requested k."""

    search_results: List[MinimalAnswer]
    k: int