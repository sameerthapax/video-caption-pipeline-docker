from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field


StyleName = Literal["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
STYLE_ORDER: tuple[StyleName, ...] = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    setting: str = ""
    subjects: list[str] = Field(default_factory=list)
    key_objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)
    audio_or_speech: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class CaptionVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_name: StyleName
    caption: str


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: str
    tone: str
    notes: str = ""


def build_observation_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "setting": {"type": "string"},
            "subjects": {"type": "array", "items": {"type": "string"}},
            "key_objects": {"type": "array", "items": {"type": "string"}},
            "actions": {"type": "array", "items": {"type": "string"}},
            "timeline": {"type": "array", "items": {"type": "string"}},
            "visible_text": {"type": "array", "items": {"type": "string"}},
            "audio_or_speech": {"type": "array", "items": {"type": "string"}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "summary",
            "setting",
            "subjects",
            "key_objects",
            "actions",
            "timeline",
            "visible_text",
            "audio_or_speech",
            "uncertainties",
        ],
    }


def build_caption_variant_json_schema(style_name: StyleName) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "style_name": {"type": "string", "enum": [style_name]},
            "caption": {"type": "string"},
        },
        "required": ["style_name", "caption"],
    }


def build_combined_captions_json_schema(styles: Sequence[StyleName]) -> dict[str, object]:
    properties = {style_name: {"type": "string"} for style_name in styles}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }


def build_judge_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "accuracy": {"type": "string", "enum": ["pass", "fail"]},
            "tone": {"type": "string", "enum": ["pass", "fail"]},
            "notes": {"type": "string"},
        },
        "required": ["accuracy", "tone", "notes"],
    }
