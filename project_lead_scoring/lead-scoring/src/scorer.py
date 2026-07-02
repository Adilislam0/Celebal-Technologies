"""
lead_scoring/src/scorer.py
X Education Lead Scoring System — Core Scoring Engine
Sellable Technologies Internship Project

Converts raw lead attributes into a 0–100 score that predicts
the probability of conversion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations (validated input values)
# ---------------------------------------------------------------------------

class LeadSource(str, Enum):
    GOOGLE = "Google"
    DIRECT_TRAFFIC = "Direct Traffic"
    ORGANIC_SEARCH = "Organic Search"
    OLARK_CHAT = "Olark Chat"
    REFERENCE = "Reference"
    WELINGAK_WEBSITE = "Welingak Website"
    FACEBOOK = "Facebook"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class Occupation(str, Enum):
    UNEMPLOYED = "Unemployed"
    STUDENT = "Student"
    WORKING_PROFESSIONAL = "Working Professional"
    BUSINESSMAN = "Businessman"
    HOUSEWIFE = "Housewife"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class LastActivity(str, Enum):
    EMAIL_OPENED = "Email Opened"
    EMAIL_BOUNCED = "Email Bounced"
    CONVERTED = "Converted to Lead"
    CHAT_CONVERSATION = "Olark Chat Conversation"
    PAGE_VISITED = "Page Visited on Website"
    EMAIL_LINK_CLICKED = "Email Link Clicked on Website"
    SMS_SENT = "SMS Sent"
    FORM_SUBMITTED = "Form Submitted on Website"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class SpecializationInterest(str, Enum):
    BUSINESS_ADMIN = "Business Administration"
    FINANCE_MANAGEMENT = "Finance Management"
    HUMAN_RESOURCE = "Human Resource Management"
    MARKETING = "Marketing Management"
    OPERATIONS = "Operations Management"
    IT_PROJECT = "IT Project Management"
    SUPPLY_CHAIN = "Supply Chain Management"
    BANKING = "Banking, Investment And Insurance"
    HOSPITALITY = "Hospitality Management"
    MEDIA = "Media and Advertising"
    TRAVEL = "Travel and Tourism"
    E_COMMERCE = "E-Commerce"
    RURAL = "Rural and Agrarian Management"
    HEALTHCARE = "Healthcare Management"
    OTHER = "Other"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Lead:
    """
    Represents a single prospective student lead.

    All fields are optional to support partial data.  Missing values are
    handled gracefully — the scorer will simply skip that dimension rather
    than fail.
    """
    # Identity
    lead_id: str = ""
    lead_number: Optional[int] = None

    # Origin
    lead_source: LeadSource = LeadSource.UNKNOWN
    lead_origin: str = ""

    # Demographics
    country: str = "India"
    city: str = ""
    occupation: Occupation = Occupation.UNKNOWN
    specialization: SpecializationInterest = SpecializationInterest.UNKNOWN

    # Engagement — quantitative
    total_visits: int = 0
    total_time_spent_on_website: int = 0          # seconds
    page_views_per_visit: float = 0.0

    # Engagement — qualitative (bool flags from CRM)
    do_not_email: bool = False
    do_not_call: bool = False
    through_recommendations: bool = False
    magazine: bool = False
    newspaper_article: bool = False
    x_education_forums: bool = False
    newspaper: bool = False
    digital_advertisement: bool = False
    through_general_digital_marketing: bool = False
    asymmetric_activities: bool = False
    last_activity: LastActivity = LastActivity.UNKNOWN
    last_notable_activity: str = ""

    # Interest signals
    what_is_your_current_occupation: str = ""
    what_matters_most_to_you_in_choosing_a_course: str = ""

    # Consent / GDPR flags
    receive_more_updates_about_our_courses: bool = False
    update_me_on_supply_chain_content: bool = False
    get_updates_on_dm_content: bool = False
    i_agree_to_pay_the_amount_through_cheque: bool = False

    # Ground truth (used during training / validation only)
    converted: Optional[bool] = None

    # Computed at score time
    score: Optional[float] = None
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    tier: Optional[str] = None


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """
    All weights, thresholds, and lookup tables for the scoring algorithm.
    Loaded from config/scoring_config.json; defaults are set here.
    """

    # Dimension max scores (must sum ≤ 100)
    weight_engagement_activity: float = 30.0
    weight_lead_origin_source: float = 20.0
    weight_demographic_fit: float = 20.0
    weight_behavioral_signals: float = 15.0
    weight_opt_in_recency: float = 15.0

    # --- Engagement ---
    max_visits_for_full_score: int = 10        # visits beyond this cap at 100%
    max_time_for_full_score: int = 2000        # seconds
    max_pvpv_for_full_score: float = 5.0       # page views per visit

    # --- Source scores (0–1) ---
    source_scores: Dict[str, float] = field(default_factory=lambda: {
        "Welingak Website": 1.0,
        "Reference": 0.9,
        "Olark Chat": 0.80,
        "Google": 0.70,
        "Organic Search": 0.65,
        "Direct Traffic": 0.60,
        "Facebook": 0.40,
        "Other": 0.30,
        "Unknown": 0.20,
    })

    # --- Occupation scores (0–1) ---
    occupation_scores: Dict[str, float] = field(default_factory=lambda: {
        "Working Professional": 1.0,
        "Businessman": 0.85,
        "Unemployed": 0.60,
        "Student": 0.50,
        "Housewife": 0.40,
        "Other": 0.35,
        "Unknown": 0.20,
    })

    # --- Last activity scores (0–1) ---
    activity_scores: Dict[str, float] = field(default_factory=lambda: {
        "Converted to Lead": 1.0,
        "Form Submitted on Website": 0.95,
        "Email Link Clicked on Website": 0.85,
        "Olark Chat Conversation": 0.80,
        "Page Visited on Website": 0.60,
        "Email Opened": 0.50,
        "SMS Sent": 0.35,
        "Email Bounced": 0.05,
        "Other": 0.25,
        "Unknown": 0.10,
    })

    # --- Tier thresholds ---
    tier_hot: float = 75.0
    tier_warm: float = 50.0
    tier_cold: float = 25.0

    @classmethod
    def from_file(cls, path: str | Path) -> "ScoringConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

class LeadScorer:
    """
    Stateless scoring engine.  Call score(lead) to get a scored copy of the
    lead with .score, .tier, and .score_breakdown populated.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or ScoringConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, lead: Lead) -> Lead:
        """Return a new Lead with .score, .tier, and .score_breakdown filled."""
        breakdown: Dict[str, float] = {}

        engagement   = self._score_engagement(lead)
        origin       = self._score_origin(lead)
        demographic  = self._score_demographic(lead)
        behavioral   = self._score_behavioral(lead)
        opt_in       = self._score_opt_in(lead)

        breakdown["engagement_activity"] = round(engagement, 2)
        breakdown["lead_origin_source"]  = round(origin, 2)
        breakdown["demographic_fit"]     = round(demographic, 2)
        breakdown["behavioral_signals"]  = round(behavioral, 2)
        breakdown["opt_in_recency"]      = round(opt_in, 2)

        total = sum(breakdown.values())
        total = max(0.0, min(100.0, total))

        tier = self._assign_tier(total)

        import copy
        result = copy.copy(lead)
        result.score = round(total, 1)
        result.score_breakdown = breakdown
        result.tier = tier
        return result

    def score_batch(self, leads: List[Lead]) -> List[Lead]:
        return [self.score(l) for l in leads]

    def explain(self, lead: Lead) -> str:
        """Human-readable explanation of the score."""
        scored = self.score(lead)
        lines = [
            f"Lead: {scored.lead_id or '(unnamed)'}",
            f"  Total Score : {scored.score:.1f} / 100  →  Tier: {scored.tier}",
            "",
            "  Dimension breakdown:",
        ]
        maxes = {
            "engagement_activity": self.cfg.weight_engagement_activity,
            "lead_origin_source":  self.cfg.weight_lead_origin_source,
            "demographic_fit":     self.cfg.weight_demographic_fit,
            "behavioral_signals":  self.cfg.weight_behavioral_signals,
            "opt_in_recency":      self.cfg.weight_opt_in_recency,
        }
        for dim, val in scored.score_breakdown.items():
            pct = (val / maxes[dim] * 100) if maxes[dim] else 0
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            lines.append(f"    {dim:<28} {val:5.1f}/{maxes[dim]:.0f}  [{bar}] {pct:.0f}%")
        lines += [
            "",
            f"  Key signals:",
            f"    Source:       {scored.lead_source.value}",
            f"    Occupation:   {scored.occupation.value}",
            f"    Last Activity:{scored.last_activity.value}",
            f"    Visits:       {scored.total_visits}",
            f"    Time on site: {scored.total_time_spent_on_website}s",
        ]
        if scored.do_not_email:
            lines.append("    ⚠ Do-not-email flag is SET")
        if scored.do_not_call:
            lines.append("    ⚠ Do-not-call flag is SET")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_engagement(self, lead: Lead) -> float:
        """
        Measures how actively the lead has engaged with the platform.
        Sub-dimensions: visits (40%), time on site (40%), page depth (20%).
        Returns a value in [0, weight_engagement_activity].
        """
        w = self.cfg.weight_engagement_activity

        # Visits (logarithmic — first few visits matter most)
        v_norm = min(lead.total_visits, self.cfg.max_visits_for_full_score) / self.cfg.max_visits_for_full_score
        visits_score = _log_scale(v_norm) * 0.40

        # Time on site (linear up to cap)
        t_norm = min(lead.total_time_spent_on_website, self.cfg.max_time_for_full_score) / self.cfg.max_time_for_full_score
        time_score = t_norm * 0.40

        # Page views per visit (linear up to cap)
        p_norm = min(lead.page_views_per_visit, self.cfg.max_pvpv_for_full_score) / self.cfg.max_pvpv_for_full_score
        pvpv_score = p_norm * 0.20

        raw = (visits_score + time_score + pvpv_score) * w
        return raw

    def _score_origin(self, lead: Lead) -> float:
        """
        High-intent sources (referrals, chat) outweigh passive ones (ads).
        Also factors in last activity quality.
        Returns a value in [0, weight_lead_origin_source].
        """
        w = self.cfg.weight_lead_origin_source

        source_val = self.cfg.source_scores.get(lead.lead_source.value, 0.20)
        activity_val = self.cfg.activity_scores.get(lead.last_activity.value, 0.10)

        # 60% source, 40% last activity
        raw = (source_val * 0.60 + activity_val * 0.40) * w
        return raw

    def _score_demographic(self, lead: Lead) -> float:
        """
        Occupation fit is the primary demographic signal from the dataset.
        Specialization match provides a secondary boost.
        Returns a value in [0, weight_demographic_fit].
        """
        w = self.cfg.weight_demographic_fit

        occ_val = self.cfg.occupation_scores.get(lead.occupation.value, 0.20)

        # Specialization: known/specific > unknown/other
        spec_val = 0.7 if (
            lead.specialization not in (SpecializationInterest.UNKNOWN, SpecializationInterest.OTHER)
        ) else 0.3

        raw = (occ_val * 0.75 + spec_val * 0.25) * w
        return raw

    def _score_behavioral(self, lead: Lead) -> float:
        """
        Positive behavioral signals (recommendations, forum activity, etc.)
        raise the score; negative signals (do-not-contact) reduce it.
        Returns a value in [0, weight_behavioral_signals].
        """
        w = self.cfg.weight_behavioral_signals

        positive_signals = [
            lead.through_recommendations,
            lead.x_education_forums,
            lead.through_general_digital_marketing,
            lead.digital_advertisement,
            lead.newspaper_article,
            lead.magazine,
        ]
        negative_signals = [
            lead.do_not_email,
            lead.do_not_call,
        ]

        pos_score = sum(positive_signals) / max(len(positive_signals), 1)
        neg_penalty = sum(negative_signals) * 0.25   # each flag costs 25% of max

        raw = max(0.0, pos_score - neg_penalty) * w
        return raw

    def _score_opt_in(self, lead: Lead) -> float:
        """
        Explicit opt-ins signal intent; asymmetric activity patterns are
        associated with lower conversion in the X Education dataset.
        Returns a value in [0, weight_opt_in_recency].
        """
        w = self.cfg.weight_opt_in_recency

        opt_ins = [
            lead.receive_more_updates_about_our_courses,
            lead.update_me_on_supply_chain_content,
            lead.get_updates_on_dm_content,
        ]
        opt_in_score = sum(opt_ins) / max(len(opt_ins), 1)

        # Asymmetric activity is a negative signal
        asym_penalty = 0.30 if lead.asymmetric_activities else 0.0

        raw = max(0.0, opt_in_score - asym_penalty) * w
        return raw

    # ------------------------------------------------------------------
    # Tier assignment
    # ------------------------------------------------------------------

    def _assign_tier(self, score: float) -> str:
        if score >= self.cfg.tier_hot:
            return "Hot"
        elif score >= self.cfg.tier_warm:
            return "Warm"
        elif score >= self.cfg.tier_cold:
            return "Cold"
        else:
            return "Disqualified"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_scale(x: float, k: float = 5.0) -> float:
    """Compresses large ranges: first gains matter most. x in [0,1]."""
    if x <= 0:
        return 0.0
    return math.log1p(k * x) / math.log1p(k)
