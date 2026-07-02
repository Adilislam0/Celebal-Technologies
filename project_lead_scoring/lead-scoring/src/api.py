"""
lead_scoring/src/api.py
Optional REST API for real-time lead scoring.
Run with: uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:
    raise SystemExit("FastAPI not installed. Run: pip install fastapi uvicorn")

from scorer import (
    Lead, LeadScorer, ScoringConfig,
    LeadSource, Occupation, LastActivity, SpecializationInterest,
)
from pipeline import score_to_json

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

CONFIG_PATH = os.environ.get("SCORING_CONFIG", "../config/scoring_config.json")
cfg = ScoringConfig.from_file(CONFIG_PATH)
scorer = LeadScorer(cfg)

app = FastAPI(
    title="X Education Lead Scoring API",
    description="Assigns conversion-probability scores to incoming leads.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class LeadRequest(BaseModel):
    lead_id: str = ""
    lead_source: str = "Unknown"
    occupation: str = "Unknown"
    last_activity: str = "Unknown"
    specialization: str = "Unknown"
    country: str = "India"
    city: str = ""
    total_visits: int = Field(0, ge=0)
    total_time_spent_on_website: int = Field(0, ge=0)
    page_views_per_visit: float = Field(0.0, ge=0)
    do_not_email: bool = False
    do_not_call: bool = False
    through_recommendations: bool = False
    magazine: bool = False
    newspaper_article: bool = False
    x_education_forums: bool = False
    digital_advertisement: bool = False
    asymmetric_activities: bool = False
    receive_more_updates_about_our_courses: bool = False
    update_me_on_supply_chain_content: bool = False
    get_updates_on_dm_content: bool = False


class ScoreResponse(BaseModel):
    lead_id: str
    score: float
    tier: str
    breakdown: Dict[str, float]
    signals: Dict


class BatchRequest(BaseModel):
    leads: List[LeadRequest]


class BatchResponse(BaseModel):
    results: List[ScoreResponse]
    summary: Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOOL_TRUES = {"yes", "true", "1"}

def _request_to_lead(req: LeadRequest) -> Lead:
    def safe_enum(cls, val, default):
        for m in cls:
            if m.value.lower() == val.strip().lower():
                return m
        return default

    return Lead(
        lead_id=req.lead_id,
        lead_source=safe_enum(LeadSource, req.lead_source, LeadSource.UNKNOWN),
        occupation=safe_enum(Occupation, req.occupation, Occupation.UNKNOWN),
        last_activity=safe_enum(LastActivity, req.last_activity, LastActivity.UNKNOWN),
        specialization=safe_enum(SpecializationInterest, req.specialization, SpecializationInterest.UNKNOWN),
        country=req.country,
        city=req.city,
        total_visits=req.total_visits,
        total_time_spent_on_website=req.total_time_spent_on_website,
        page_views_per_visit=req.page_views_per_visit,
        do_not_email=req.do_not_email,
        do_not_call=req.do_not_call,
        through_recommendations=req.through_recommendations,
        magazine=req.magazine,
        newspaper_article=req.newspaper_article,
        x_education_forums=req.x_education_forums,
        digital_advertisement=req.digital_advertisement,
        asymmetric_activities=req.asymmetric_activities,
        receive_more_updates_about_our_courses=req.receive_more_updates_about_our_courses,
        update_me_on_supply_chain_content=req.update_me_on_supply_chain_content,
        get_updates_on_dm_content=req.get_updates_on_dm_content,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": "rule-based-v1"}


@app.post("/score", response_model=ScoreResponse)
def score_lead(req: LeadRequest):
    lead = _request_to_lead(req)
    result = scorer.score(lead)
    return ScoreResponse(**score_to_json(result))


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest):
    if len(req.leads) > 5000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 5000.")
    leads = [_request_to_lead(r) for r in req.leads]
    scored = scorer.score_batch(leads)
    results = [ScoreResponse(**score_to_json(s)) for s in scored]

    from collections import Counter
    tier_counts = Counter(s.tier for s in scored)
    avg_score   = sum(s.score for s in scored) / len(scored) if scored else 0

    return BatchResponse(
        results=results,
        summary={
            "total": len(scored),
            "avg_score": round(avg_score, 1),
            "tier_counts": dict(tier_counts),
        },
    )


@app.get("/config")
def get_config():
    return cfg.__dict__
