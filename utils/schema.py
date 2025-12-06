from pydantic import BaseModel, field_validator, Field
from typing import List, Union, ClassVar
from dataclasses import dataclass
from datetime import datetime

class DeleteRequest(BaseModel):
    filename: Union[str, List[str]]
    task: str

DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

class ClassificationFeatures(BaseModel):
    tracking_status: int = Field(..., description="Tracking Status dalam int")
    tracking_first_opened_at: datetime = Field(..., description="Tracking Opened At dalam bentuk Date (5/30/2018 14:54:33)")
    tracking_completed_at: datetime = Field(..., description="Tracking Completed At dalam bentuk Date (5/30/2018 14:54:33)")

    completion_created_at: datetime = Field(..., description="Completion Created At")
    completion_enrolling_times: float = Field(..., description="Completion Enrolling dalam Float")
    completion_study_duration: float = Field(..., description="Completion Study Duration dalam Float")
    completion_avg_submission_rating: float = Field(..., description="Completion Avg Rating dalam Float")

    submission_status: float = Field(..., description="Submission Status dalam Float")
    submission_created_at: datetime = Field(..., description="Submission Created At Date (5/30/2018 14:54:33)")
    submission_duration: float = Field(..., description="Submission Duration dalam Float")
    submission_ended_review_at: datetime = Field(..., description="Submission End Review Date (5/30/2018 14:54:33)")
    submission_rating: float = Field(..., description="Submission Rating dalam Float")

    # daftar field datetime
    datetime_fields: ClassVar[set] = {
        "tracking_first_opened_at",
        "tracking_completed_at",
        "completion_created_at",
        "submission_created_at",
        "submission_ended_review_at",
    }

    @field_validator("*", mode="before")
    def parse_dates(cls, v, info):
        if info.field_name in cls.datetime_fields and isinstance(v, str):
            return datetime.strptime(v, DATE_FORMAT)
        return v