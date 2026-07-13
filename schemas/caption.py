from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


StyleName = Literal["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
STYLE_ORDER: tuple[StyleName, ...] = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


class TimelineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str = ""
    observation: str = ""


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_summary: str = ""
    setting: str = ""
    subjects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    key_objects: list[str] = Field(default_factory=list)
    timeline: list[TimelineObservation] = Field(default_factory=list)
    temporal_highlights: list[str] = Field(default_factory=list)
    camera: str = ""
    caption_facts: list[str] = Field(default_factory=list)


class CaptionVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_name: StyleName
    caption: str


class CaptionCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_1: str
    candidate_2: str


class CandidateJudgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: str
    style: str
    accuracy_score: float = Field(ge=0.0, le=1.0)
    style_score: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_candidate: str
    candidate_1: CandidateJudgeEvaluation
    candidate_2: CandidateJudgeEvaluation


def build_observation_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scene_summary": {"type": "string"},
            "setting": {"type": "string"},
            "subjects": {"type": "array", "items": {"type": "string"}},
            "actions": {"type": "array", "items": {"type": "string"}},
            "key_objects": {"type": "array", "items": {"type": "string"}},
            "timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "timestamp": {"type": "string"},
                        "observation": {"type": "string"},
                    },
                    "required": ["timestamp", "observation"],
                },
            },
            "temporal_highlights": {"type": "array", "items": {"type": "string"}},
            "camera": {"type": "string"},
            "caption_facts": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "scene_summary",
            "setting",
            "subjects",
            "actions",
            "key_objects",
            "timeline",
            "temporal_highlights",
            "camera",
            "caption_facts",
        ],
    }


def build_caption_candidates_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_1": {"type": "string"},
            "candidate_2": {"type": "string"},
        },
        "required": ["candidate_1", "candidate_2"],
    }


def build_judge_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_candidate": {"type": "string", "enum": ["candidate_1", "candidate_2"]},
            "candidate_1": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "accuracy": {"type": "string", "enum": ["pass", "fail"]},
                    "style": {"type": "string", "enum": ["pass", "fail"]},
                    "accuracy_score": {"type": "number"},
                    "style_score": {"type": "number"},
                    "combined_score": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["accuracy", "style", "accuracy_score", "style_score", "combined_score", "notes"],
            },
            "candidate_2": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "accuracy": {"type": "string", "enum": ["pass", "fail"]},
                    "style": {"type": "string", "enum": ["pass", "fail"]},
                    "accuracy_score": {"type": "number"},
                    "style_score": {"type": "number"},
                    "combined_score": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["accuracy", "style", "accuracy_score", "style_score", "combined_score", "notes"],
            },
        },
        "required": ["selected_candidate", "candidate_1", "candidate_2"],
    }
