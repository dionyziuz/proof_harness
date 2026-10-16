"""Pydantic Models and Schemas for Prime Math Web Portal."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ProblemSubmission(BaseModel):
    """Mathematician problem submission payload."""
    problem_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique URL-friendly slug identifier for the problem (e.g. erdos-distinct-distances)",
    )
    title: str = Field(
        ...,
        min_length=5,
        max_length=256,
        description="Descriptive mathematical title",
    )
    field: str = Field(
        default="Combinatorics & Number Theory",
        max_length=128,
        description="Subfield of mathematics (e.g., Number Theory, TCS, Additive Combinatorics)",
    )
    context: Optional[str] = Field(
        default="",
        max_length=40000,
        description="Definitions, notations, preliminary lemmas, and established literature context (LaTeX supported)",
    )
    statement: str = Field(
        ...,
        min_length=10,
        max_length=20000,
        description="Target theorem, conjecture, or reduction to prove or refute (LaTeX supported)",
    )
    hypotheses: Optional[List[str]] = Field(
        default_factory=list,
        description="Explicit hypotheses, invariant preconditions, or known boundary constraints",
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Domain tags (e.g. ['Algebraic', 'Extremal', 'P-NP', 'Spectral'])",
    )
    notability: Optional[str] = Field(
        default="Open Research Problem",
        description="Problem source or notability tier",
    )
    literature_search: bool = Field(
        default=False,
        description="Whether to allow live mathematical literature search during proving (TCS-Bench protocol defaults to False)",
    )
    max_iterations: Optional[int] = Field(
        default=8,
        ge=1,
        le=16,
        description="Maximum exploratory reasoning turns (1-16)",
    )

    @field_validator("problem_id")
    @classmethod
    def validate_problem_id(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.match(r"^[a-z0-9_\-\.]+$", v):
            raise ValueError("problem_id must consist only of lowercase alphanumeric characters, dashes, underscores, and dots.")
        return v

    @field_validator("statement", "title")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text fields cannot be empty or whitespace only.")
        return v

    def to_formatted_prompt(self) -> str:
        """Formats the problem submission into a self-contained solver prompt."""
        parts: List[str] = []
        parts.append(f"# Problem: {self.title}\n")
        parts.append(f"**Mathematical Field**: {self.field}\n")
        
        if self.tags:
            parts.append(f"**Tags**: {', '.join(self.tags)}\n")
            
        if self.context and self.context.strip():
            parts.append("## Definitions, Conventions, and Context\n" + self.context.strip() + "\n")
            
        if self.hypotheses:
            valid_hyps = [h.strip() for h in self.hypotheses if h.strip()]
            if valid_hyps:
                parts.append("## Hypotheses & Invariants\n" + "\n".join(f"- {h}" for h in valid_hyps) + "\n")
                
        parts.append("## Target Statement to Prove\n" + self.statement.strip() + "\n")
        
        parts.append(
            "\n## Prover Instructions\n"
            "1. Output your complete formal LaTeX proof enclosed strictly within \\begin{proof} ... \\end{proof}.\n"
            "2. Establish all intermediate lemmas from first principles using verified invariants.\n"
            "3. Ensure all boundary cases and parameter inequalities are fully addressed.\n"
        )
        return "\n".join(parts)


class JobRecord(BaseModel):
    """Full database record for an active or completed proving job."""
    id: str
    problem_id: str
    title: str
    field: str
    user_email: str
    status: str  # 'queued', 'held_operator_priority', 'running', 'completed', 'failed', 'cancelled'
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    priority: int = 0
    problem_json: Dict[str, Any]
    task_prompt: str
    iterations: int = 0
    max_iterations: int = 8
    duration_s: float = 0.0
    validity: Optional[float] = None
    candidate_proof: Optional[str] = None
    log_path: Optional[str] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None


class JobSummary(BaseModel):
    """Lightweight summary of a job for lists and status views."""
    id: str
    problem_id: str
    title: str
    field: str
    user_email: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    iterations: int = 0
    max_iterations: int = 8
    duration_s: float = 0.0
    validity: Optional[float] = None
    queue_position: Optional[int] = None
    has_proof: bool = False


class QueueSettings(BaseModel):
    """Live queue settings and status."""
    operator_lock: bool
    operator_lock_reason: str
    max_workers: int
    active_workers: int
    queued_count: int
    held_count: int
    running_count: int
    completed_count: int


class OperatorLockToggle(BaseModel):
    """Request to change operator lock status."""
    locked: bool
    reason: Optional[str] = "Operator evaluation running on local APU workers"


class InvitationCode(BaseModel):
    """Invitation key record."""
    code: str
    created_at: str
    created_by: str = "admin"
    used_by: Optional[str] = None
    used_at: Optional[str] = None
    is_active: bool = True
    notes: Optional[str] = ""


class InvitationCreateRequest(BaseModel):
    notes: Optional[str] = "Invited research mathematician"


class AuthVerifyRequest(BaseModel):
    email: str
    invite_code: str
    password: Optional[str] = None
    magic_token: Optional[str] = None
