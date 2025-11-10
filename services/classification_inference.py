from fastapi import HTTPException
from dotenv import load_dotenv
import os
import numpy as np

import joblib

class ClassificationService:
    def __init__(self):
        load_dotenv()
        model_path = os.getenv("MODEL_PATH", "unknown.pt")

    async def analysis(self):
        pass