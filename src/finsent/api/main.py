"""FastAPI app serving the champion sentiment model."""

import json
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from transformers import pipeline

from finsent.config import MODEL_DIR
from finsent.db.engine import engine
from finsent.db.queries import save_prediction

logger = logging.getLogger(__name__)
MAX_LENGTH = 64  # same limit as training


# ---------- Schemas ----------
class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=1000,
        examples=["$AAPL beats estimates, raises guidance"],
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    prediction_id: int | None = None


# ---------- Startup: load the model once ----------
def load_classifier():
    """Blocking work: load the model and read its version label."""
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=str(MODEL_DIR), device=device)
    with open(MODEL_DIR / "model_info.json") as f:
        info = json.load(f)
    return classifier, f"{info['name']}:v{info['version']}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier, app.state.model_version = load_classifier()
    yield

app = FastAPI(title="finsent", lifespan=lifespan)


# ---------- Helpers ----------
def database_ok() -> bool:
    """True if Postgres answers a trivial query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


# ---------- Endpoints ----------
@app.get("/health")
def health():
    loaded = getattr(app.state, "classifier", None) is not None
    return {
        "status": "ok" if loaded else "unavailable",  # based on the model only
        "model_loaded": loaded,
        "model_version": getattr(app.state, "model_version", None),
        "database": "ok" if database_ok() else "unavailable",  # informational
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    result = app.state.classifier(req.text, truncation=True, max_length=MAX_LENGTH)[0]
    label, confidence = result["label"], float(result["score"])

    # Store the prediction; if the DB is down, still answer (option B)
    prediction_id = None
    try:
        prediction_id = save_prediction(
            engine,
            input_text=req.text,
            predicted_label=label,
            confidence=confidence,
            model_version=app.state.model_version,
        )
    except SQLAlchemyError:
        logger.exception("Could not save prediction to the database")

    return PredictResponse(
        label=label,
        confidence=confidence,
        model_version=app.state.model_version,
        prediction_id=prediction_id,
    )
