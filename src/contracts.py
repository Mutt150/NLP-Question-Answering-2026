"""Bentuk data bersama. Semua tipe di bawah berupa dict biasa saat dijalankan."""

from typing import Literal, TypedDict


AnswerType = Literal["PERSON", "DATE", "LOCATION", "NUMBER", "ORGANIZATION", "FACT", "OTHER"]


class CorpusPassage(TypedDict):
    passage_id: str
    doc_id: str
    title: str
    text: str
    source: str


class QuestionInfo(TypedDict):
    answer_type: AnswerType
    keywords: list[str]
    relation_cues: list[str]


class RetrievedPassage(TypedDict):
    passage_id: str
    text: str
    score: float
    source: str


class Answer(TypedDict):
    status: Literal["answered", "needs_review", "not_found"]
    answer: str | None
    passage_id: str | None
    evidence: str | None
    note: str | None


class QAResult(Answer):
    question: str
    source: str | None
    passage: RetrievedPassage | None


class EvaluationRow(TypedDict):
    question_id: str
    question: str
    ground_truth: str
    classic_answer: str | None
    classic_passage_id: str | None
    classic_correct: bool | None
    rag_answer: str | None
    rag_passage_id: str | None
    rag_correct: bool | None
    error_reason: str | None


class UnavailableError(RuntimeError):
    """Komponen, model, atau layanan yang diperlukan belum dapat digunakan."""
