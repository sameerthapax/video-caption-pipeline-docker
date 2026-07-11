from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field


StyleName = Literal["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
STYLE_ORDER: tuple[StyleName, ...] = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class Setting(BaseModel):
        model_config = ConfigDict(extra="forbid")

        location_type: str = ""
        environment: str = ""
        time_of_day: str = ""
        weather: str = ""

    class Subject(BaseModel):
        model_config = ConfigDict(extra="forbid")

        type: str = ""
        count: str = ""
        description: str = ""

    class KeyObject(BaseModel):
        model_config = ConfigDict(extra="forbid")

        object: str = ""
        description: str = ""

    class Timeline(BaseModel):
        model_config = ConfigDict(extra="forbid")

        beginning: str = ""
        middle: str = ""
        end: str = ""

    class Camera(BaseModel):
        model_config = ConfigDict(extra="forbid")

        viewpoint: str = ""
        movement: str = ""

    summary: str = ""
    setting: Setting = Field(default_factory=Setting)
    subjects: list[Subject] = Field(default_factory=list)
    key_objects: list[KeyObject] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    timeline: Timeline = Field(default_factory=Timeline)
    camera: Camera = Field(default_factory=Camera)
    visible_text: list[str] = Field(default_factory=list)
    audio_or_speech: list[str] = Field(default_factory=list)
    distinctive_details: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class CaptionVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_name: StyleName
    caption: str
    grounded_facts_used: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: float = Field(ge=0.0, le=1.0)
    style_match: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""


def build_observation_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "setting": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "location_type": {"type": "string"},
                    "environment": {"type": "string"},
                    "time_of_day": {"type": "string"},
                    "weather": {"type": "string"},
                },
                "required": ["location_type", "environment", "time_of_day", "weather"],
            },
            "subjects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string"},
                        "count": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["type", "count", "description"],
                },
            },
            "key_objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "object": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["object", "description"],
                },
            },
            "relationships": {"type": "array", "items": {"type": "string"}},
            "actions": {"type": "array", "items": {"type": "string"}},
            "timeline": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "beginning": {"type": "string"},
                    "middle": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["beginning", "middle", "end"],
            },
            "camera": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "viewpoint": {"type": "string"},
                    "movement": {"type": "string"},
                },
                "required": ["viewpoint", "movement"],
            },
            "visible_text": {"type": "array", "items": {"type": "string"}},
            "audio_or_speech": {"type": "array", "items": {"type": "string"}},
            "distinctive_details": {"type": "array", "items": {"type": "string"}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "summary",
            "setting",
            "subjects",
            "key_objects",
            "relationships",
            "actions",
            "timeline",
            "camera",
            "visible_text",
            "audio_or_speech",
            "distinctive_details",
            "uncertainties",
        ],
    }


def build_caption_variant_json_schema(style_name: StyleName) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "style_name": {"type": "string", "enum": [style_name]},
            "caption": {"type": "string"},
            "grounded_facts_used": {"type": "array", "items": {"type": "string"}},
            "safety_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["style_name", "caption", "grounded_facts_used", "safety_notes"],
    }


def build_combined_captions_json_schema(styles: Sequence[StyleName]) -> dict:
    properties = {
        style_name: build_caption_variant_json_schema(style_name)
        for style_name in styles
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }


def build_judge_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
            "style_match": {"type": "number", "minimum": 0, "maximum": 1},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "feedback": {"type": "string"},
        },
        "required": ["accuracy", "style_match", "score", "feedback"],
    }
