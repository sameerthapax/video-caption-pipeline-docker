from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class CaptionTask(BaseModel):
    task_id: str = Field(min_length=1)
    video_url: HttpUrl
    styles: list[str] = Field(default_factory=list)
    transcript_url: HttpUrl | None = None
    transcript_text: str = ""


class TaskResult(BaseModel):
    task_id: str
    captions: dict[str, str]
