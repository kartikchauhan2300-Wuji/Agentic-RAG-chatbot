"""schemas.py — data contracts for the research-paper agent.

Models passed to the LLM (with_structured_output): PaperSummary, RouteDecision, GradeDocuments.
Containers filled by code: ComparisonEntry, PaperComparison, Citation.
Order matters: ComparisonEntry must be defined before PaperComparison.
"""
from typing import Literal

from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """Structured extraction of a key finding from a research paper."""

    dataset_used: str | None = Field(
        default=None,
        description="The primary dataset used in the paper's experiments, if mentioned.",
    )
    methodology: str = Field(
        description="A concise description of the core method or approach proposed."
    )
    key_result: str = Field(
        description="The main quantitative or qualitative result reported."
    )
    source_section: Literal["abstract", "methodology", "results", "conclusion", "other"] = Field(
        description="Which section of the paper this information was extracted from."
    )


class RouteDecision(BaseModel):
    """Classify what kind of question the user is asking."""

    route: Literal["qa", "extract", "compare"] = Field(
        description="'qa' = question about one paper, 'extract' = pull structured "
        "facts, 'compare' = contrast several papers."
    )


class GradeDocuments(BaseModel):
    """Binary relevance grade for retrieved text."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the text is relevant to the question, otherwise 'no'."
    )


class ComparisonEntry(BaseModel):
    """One paper's finding on the aspect being compared (source/page set by code)."""

    source: str = Field(description="Paper identifier or filename.")
    finding: str = Field(description="What this paper says about the aspect.")
    page: int | None = Field(default=None, description="Page the finding came from.")


class PaperComparison(BaseModel):
    """Side-by-side comparison of one aspect across papers."""

    aspect: str = Field(description="What is compared, e.g. 'methodology'.")
    entries: list[ComparisonEntry]
    summary: str = Field(description="Two sentences on how the papers differ.")


class Citation(BaseModel):
    """Filled from chunk metadata by code, never by the model."""

    source: str
    page: int | None
    heading: str


