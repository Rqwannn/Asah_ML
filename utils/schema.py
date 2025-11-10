from pydantic import BaseModel
from typing import List, Union
from dataclasses import dataclass

class DeleteRequest(BaseModel):
    filename: Union[str, List[str]]
    task: str

from pydantic import BaseModel, Field
from typing import Literal, Optional

class ClassificationFeatures(BaseModel):
    pass