"""
lead_scoring/src/pipeline.py
Ingestion pipeline: CSV → Lead objects → scored output
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional

from scorer import (
    Lead, LeadScorer, ScoringConfig,
    LeadSource, Occupation, LastActivity, SpecializationInterest,
)


# ---------------------------------------------------------------------------
# CSV → Lead
# ---------------------------------------------------------------------------

# Maps CSV column names (lower-cased, stripped) to Lead field names
_COLUMN_MAP = {
    "prospect id":                     "lead_id",
    "lead number":                     "lead_number",
    "lead origin":                     "lead_origin",
    "lead source":                     "lead_source",
    "do not email":                    "do_not_email",
    "do not call":                     "do_not_call",
    "totalvisits":                     "total_visits",
    "total time spent on website":     "total_time_spent_on_website",
    "page views per visit":            "page_views_per_visit",
    "last activity":                   "last_activity",
    "country":                         "country",
    "specialization":                  "specialization",
    "what is your current occupation": "occupation",
    "what matters most to you in choosing a course": "what_matters_most_to_you_in_choosing_a_course",
    "x education forums":              "x_education_forums",
    "newspaper article":               "newspaper_article",
    "digital advertisement":           "digital_advertisement",
    "through recommendations":         "through_recommendations",
    "receive more updates about our courses": "receive_more_updates_about_our_courses",
    "update me on supply chain content": "update_me_on_supply_chain_content",
    "get updates on dm content":       "get_updates_on_dm_content",
    "city":                            "city",
    "asymmetrique activity index":     "asymmetric_activities",
    "magazine":                        "magazine",
    "newspaper":                       "newspaper",
    "converted":                       "converted",
    "last notable activity":           "last_notable_activity",
}

_BOOL_TRUES = {"yes", "true", "1", "y"}


def _safe_enum(enum_cls, raw: str, default):
    raw = raw.strip()
    for member in enum_cls:
        if member.value.lower() == raw.lower():
            return member
    return default


def _parse_row(row: dict) -> Lead:
    normalised = {k.strip().lower(): v.strip() for k, v in row.items()}
    lead = Lead()

    for csv_col, field_name in _COLUMN_MAP.items():
        raw = normalised.get(csv_col, "")
        if not raw or raw.lower() in ("nan", "select", "not provided", ""):
            continue

        if field_name == "lead_source":
            lead.lead_source = _safe_enum(LeadSource, raw, LeadSource.UNKNOWN)
        elif field_name == "last_activity":
            lead.last_activity = _safe_enum(LastActivity, raw, LastActivity.UNKNOWN)
        elif field_name == "occupation":
            lead.occupation = _safe_enum(Occupation, raw, Occupation.UNKNOWN)
        elif field_name == "specialization":
            lead.specialization = _safe_enum(SpecializationInterest, raw, SpecializationInterest.UNKNOWN)
        elif field_name in ("do_not_email", "do_not_call", "through_recommendations",
                             "magazine", "newspaper_article", "x_education_forums",
                             "newspaper", "digital_advertisement",
                             "through_general_digital_marketing", "asymmetric_activities",
                             "receive_more_updates_about_our_courses",
                             "update_me_on_supply_chain_content",
                             "get_updates_on_dm_content"):
            setattr(lead, field_name, raw.lower() in _BOOL_TRUES)
        elif field_name == "converted":
            lead.converted = raw.lower() in _BOOL_TRUES or raw == "1"
        elif field_name in ("total_visits", "lead_number"):
            try:
                setattr(lead, field_name, int(float(raw)))
            except ValueError:
                pass
        elif field_name in ("total_time_spent_on_website", "page_views_per_visit"):
            try:
                setattr(lead, field_name, float(raw))
            except ValueError:
                pass
        else:
            setattr(lead, field_name, raw)

    return lead


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> List[Lead]:
    p = Path(path)
    leads: List[Lead] = []
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leads.append(_parse_row(row))
    return leads


def score_csv(
    input_path: str | Path,
    output_path: str | Path,
    config_path: Optional[str | Path] = None,
) -> None:
    """
    Read leads from a CSV, score them, and write an enriched CSV with
    Score, Tier, and dimension columns appended.
    """
    cfg = ScoringConfig.from_file(config_path) if config_path else ScoringConfig()
    scorer = LeadScorer(cfg)

    leads = load_csv(input_path)
    scored = scorer.score_batch(leads)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dimension_cols = [
        "score", "tier",
        "dim_engagement", "dim_origin", "dim_demographic",
        "dim_behavioral", "dim_opt_in",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "lead_id", "lead_source", "occupation", "last_activity",
            "total_visits", "total_time_spent_on_website", "page_views_per_visit",
            "do_not_email", "do_not_call", "converted",
        ] + dimension_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for l in scored:
            writer.writerow({
                "lead_id":   l.lead_id,
                "lead_source": l.lead_source.value,
                "occupation": l.occupation.value,
                "last_activity": l.last_activity.value,
                "total_visits": l.total_visits,
                "total_time_spent_on_website": l.total_time_spent_on_website,
                "page_views_per_visit": l.page_views_per_visit,
                "do_not_email": l.do_not_email,
                "do_not_call": l.do_not_call,
                "converted": l.converted,
                "score": l.score,
                "tier": l.tier,
                "dim_engagement":  l.score_breakdown.get("engagement_activity", 0),
                "dim_origin":      l.score_breakdown.get("lead_origin_source", 0),
                "dim_demographic": l.score_breakdown.get("demographic_fit", 0),
                "dim_behavioral":  l.score_breakdown.get("behavioral_signals", 0),
                "dim_opt_in":      l.score_breakdown.get("opt_in_recency", 0),
            })

    print(f"✓ Scored {len(scored)} leads → {out}")

    # Print tier summary
    from collections import Counter
    tiers = Counter(l.tier for l in scored)
    print("\nTier distribution:")
    for t in ["Hot", "Warm", "Cold", "Disqualified"]:
        n = tiers.get(t, 0)
        pct = n / len(scored) * 100 if scored else 0
        print(f"  {t:<15} {n:>5}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# JSON output helper (for API integration)
# ---------------------------------------------------------------------------

def score_to_json(lead: Lead) -> dict:
    return {
        "lead_id": lead.lead_id,
        "score": lead.score,
        "tier": lead.tier,
        "breakdown": lead.score_breakdown,
        "signals": {
            "lead_source": lead.lead_source.value,
            "occupation": lead.occupation.value,
            "last_activity": lead.last_activity.value,
            "total_visits": lead.total_visits,
            "time_on_site_seconds": lead.total_time_spent_on_website,
            "page_views_per_visit": lead.page_views_per_visit,
            "do_not_email": lead.do_not_email,
            "do_not_call": lead.do_not_call,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="X Education Lead Scoring Pipeline")
    parser.add_argument("input",  help="Input CSV path (leads)")
    parser.add_argument("output", help="Output CSV path (scored leads)")
    parser.add_argument("--config", default=None, help="Path to scoring_config.json")
    args = parser.parse_args()

    score_csv(args.input, args.output, args.config)
