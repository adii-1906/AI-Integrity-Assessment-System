import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    LOCAL_MODEL_NAME = "google/flan-t5-base"
    WEIGHTS = {
        "m1_bias": 0.20,
        "m2_hallucination": 0.25,
        "m3_privacy": 0.20,
        "m4_explainability": 0.15,
        "m5_deepfake": 0.20
    }
    BIAS_HIGH_THRESHOLD = 0.6
    BIAS_MILD_THRESHOLD = 0.3
    HALLUCINATION_THRESHOLD = 0.5
    PRIVACY_HIGH_RISK_THRESHOLD = 0.7
    KNOWLEDGE_BASE_PATH = "data/knowledge_base.json"

    @property
    def has_gemini(self):
        return bool(self.GEMINI_API_KEY)

    @property
    def has_perspective(self):
        return bool(self.PERSPECTIVE_API_KEY)

config = Config()