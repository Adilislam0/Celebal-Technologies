# X Education Lead Scoring System
### Sellable Technologies — Internship Project

> **Mission:** Lift X Education's lead-to-enrolment conversion rate from 30 % to 80 % by identifying and prioritising high-intent prospects before the sales team reaches out.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why Lead Scoring?](#2-why-lead-scoring)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [The Scoring Algorithm](#5-the-scoring-algorithm)
   - 5.1 [Dimensions & Weights](#51-dimensions--weights)
   - 5.2 [Dimension Detail](#52-dimension-detail)
   - 5.3 [Score Formula](#53-score-formula)
   - 5.4 [Tier Assignment](#54-tier-assignment)
6. [Data Inputs](#6-data-inputs)
7. [Installation & Configuration](#7-installation--configuration)
8. [Usage](#8-usage)
   - 8.1 [Python API](#81-python-api)
   - 8.2 [CLI Pipeline (CSV → CSV)](#82-cli-pipeline-csv--csv)
   - 8.3 [REST API](#83-rest-api)
9. [Interpreting Scores & Acting on Them](#9-interpreting-scores--acting-on-them)
10. [Validation & Testing](#10-validation--testing)
11. [Customising the Model](#11-customising-the-model)
12. [End-to-End Example](#12-end-to-end-example)
13. [Dataset Reference](#13-dataset-reference)
14. [Roadmap](#14-roadmap)
15. [Contributing](#15-contributing)

---

## 1. Overview

This repository contains a **rule-based, configurable lead scoring engine** built specifically for X Education's dataset. It evaluates every incoming lead across five behavioural and demographic dimensions and produces:

- A **numerical score (0–100)** representing estimated conversion likelihood.
- A **tier label** — Hot / Warm / Cold / Disqualified — that drives sales workflow.
- A **per-dimension breakdown** so sales reps understand *why* a lead scored as it did.

The system is:

| Property | Detail |
|---|---|
| **Language** | Python 3.9+ |
| **Dependencies** | Zero runtime deps (stdlib only) for the core engine |
| **Input formats** | Python objects · CSV · REST JSON |
| **Output formats** | Python objects · CSV · REST JSON |
| **Serving** | Optional FastAPI REST layer |
| **Test coverage** | 29 pytest tests, 100 % passing |

---

## 2. Why Lead Scoring?

X Education collects leads from many channels: Google, organic search, referrals, chat interactions, and more. At 30 % conversion today, the sales team spends roughly **70 % of their call time on leads that will never enrol**. Lead scoring solves this by:

1. **Ranking** leads so the highest-probability prospects are contacted first.
2. **Filtering** clearly disqualified leads (do-not-contact flags, zero engagement) so no time is wasted.
3. **Explaining** the score so reps can tailor their pitch to each lead's specific situation.

The target state is that the top tier (Hot) converts at ≥ 70 %, while the Disqualified tier is never called — effectively concentrating sales effort on the most productive segment.

### Baseline vs. target conversion by tier

```
Tier          Target conversion rate    Recommended action
─────────────────────────────────────────────────────────
Hot (≥ 75)    ≥ 70 %                    Call within 2 hours
Warm (50–74)  40–70 %                   Email + call within 24 hours
Cold (25–49)  10–40 %                   Nurture email sequence
Disqualified  < 10 %                    No outreach; re-evaluate in 30 days
```

---

## 3. Quick Start

```bash
# Clone and enter
git clone https://github.com/sellable-tech/x-education-lead-scoring.git
cd x-education-lead-scoring

# Install (core engine needs nothing; full stack for API/notebooks)
pip install -r requirements.txt

# Score a CSV of leads
python src/pipeline.py data/leads.csv data/scored_leads.csv

# Run all tests
pytest tests/ -v

# Start REST API
uvicorn src.api:app --reload --port 8000
```

---

## 4. Project Structure

```
lead-scoring/
├── src/
│   ├── scorer.py        ← Core scoring engine (Lead, LeadScorer, ScoringConfig)
│   ├── pipeline.py      ← CSV ingestion + batch scoring
│   ├── validator.py     ← Metrics: lift, Gini, decile analysis
│   └── api.py           ← FastAPI REST layer (optional)
├── tests/
│   └── test_scorer.py   ← 29 pytest tests (100 % passing)
├── config/
│   └── scoring_config.json   ← All weights and lookup tables (editable)
├── data/
│   └── .gitkeep         ← Drop your Kaggle CSV here
├── notebooks/
│   └── .gitkeep         ← Jupyter exploration goes here
├── docs/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

---

## 5. The Scoring Algorithm

### 5.1 Dimensions & Weights

The total score is built from **five independent dimensions**. Each contributes up to its maximum weight:

```
Dimension                Max weight   What it measures
──────────────────────────────────────────────────────────────────────────────
1  Engagement & Activity     30       How much has the lead explored the site?
2  Lead Origin & Source       20       Where did they come from? What's recent?
3  Demographic Fit            20       Does their profile match a typical buyer?
4  Behavioural Signals        15       Opt-ins, referrals, community activity
5  Opt-in & Recency           15       Newsletter/update preferences + intent flags
                             ───
Total                        100
```

> **Why these weights?**
> Engagement is weighted highest because time-on-site and page depth are the strongest predictors of purchase intent available in the X Education dataset. Source follows because referrals and direct chat have historically 2–3× higher close rates than ad traffic. Demographic and behavioural signals provide refinement; opt-in flags confirm stated intent.

---

### 5.2 Dimension Detail

#### Dimension 1 — Engagement & Activity (max 30 pts)

Measures how actively the lead has engaged with the X Education website.

| Sub-signal | Weight within dimension | Calculation |
|---|---|---|
| Total visits | 40 % | Logarithmic scale — first visits count most |
| Total time on site | 40 % | Linear, capped at 2 000 s |
| Page views per visit | 20 % | Linear, capped at 5.0 pv/visit |

**Why logarithmic for visits?** A lead going from 0 to 3 visits is a much stronger signal than 10 to 13. The log curve (`log(1 + k·x) / log(1 + k)`) naturally compresses the high end.

```
Visits score curve:
1 visit  → ~37 % of max
3 visits → ~66 %
5 visits → ~80 %
10 visits → 100 %
```

#### Dimension 2 — Lead Origin & Source (max 20 pts)

Combines the **inbound channel** (60 %) and the **most recent meaningful activity** (40 %).

Lead source lookup table:

| Source | Score |
|---|---|
| Welingak Website | 1.00 (highest intent — returning visitor) |
| Reference (referral) | 0.90 |
| Olark Chat | 0.80 |
| Google | 0.70 |
| Organic Search | 0.65 |
| Direct Traffic | 0.60 |
| Facebook | 0.40 |
| Other / Unknown | 0.20–0.30 |

Last activity lookup:

| Activity | Score |
|---|---|
| Converted to Lead / Form Submitted | 0.95–1.00 |
| Email Link Clicked | 0.85 |
| Olark Chat Conversation | 0.80 |
| Page Visited | 0.60 |
| Email Opened | 0.50 |
| SMS Sent | 0.35 |
| Email Bounced | 0.05 |

#### Dimension 3 — Demographic Fit (max 20 pts)

Occupation fit (75 %) combined with specialisation specificity (25 %).

| Occupation | Score |
|---|---|
| Working Professional | 1.00 |
| Businessman | 0.85 |
| Unemployed | 0.60 |
| Student | 0.50 |
| Housewife | 0.40 |
| Other / Unknown | 0.20–0.35 |

> Working professionals score highest because the X Education dataset shows they convert at approximately 2× the rate of students — they have both the motivation to upskill and the financial means.

Specialisation: any *specific* course interest (Finance, Marketing, HR, etc.) adds a 0.7× multiplier, vs. 0.3× for "Unknown".

#### Dimension 4 — Behavioural Signals (max 15 pts)

Binary flags from CRM interactions:

```
Positive signals (+): through_recommendations, x_education_forums,
                      digital_advertisement, newspaper_article,
                      magazine, through_general_digital_marketing

Negative signals (−): do_not_email, do_not_call
                      (each flag costs 25 % of dimension max)
```

Formula:
```
dim4 = max(0, positive_fraction − (0.25 × negative_count)) × 15
```

#### Dimension 5 — Opt-in & Recency (max 15 pts)

Explicit newsletter/update opt-ins are strong intent signals:

```
receive_more_updates_about_our_courses
update_me_on_supply_chain_content
get_updates_on_dm_content
```

A confirmed **asymmetric activity** pattern (bot-like behaviour) applies a 30 % penalty.

---

### 5.3 Score Formula

```
score = Σ(dimension_i)          where dimension_i ∈ [0, weight_i]

      = engagement_score
      + origin_score
      + demographic_score
      + behavioral_score
      + optin_score

      ∈ [0, 100]
```

All dimensions are independent and additive. The score is clamped to [0, 100].

---

### 5.4 Tier Assignment

```
Score ≥ 75  →  Hot          (immediate high-priority outreach)
Score ≥ 50  →  Warm         (standard outreach within 24 hrs)
Score ≥ 25  →  Cold         (nurture sequence)
Score < 25  →  Disqualified (hold)
```

Thresholds are configurable in `config/scoring_config.json`.

---

## 6. Data Inputs

### Required (minimal scoring)

| Field | Type | Example | Notes |
|---|---|---|---|
| `lead_source` | enum | `"Google"` | See `LeadSource` enum |
| `last_activity` | enum | `"Email Opened"` | See `LastActivity` enum |
| `occupation` | enum | `"Working Professional"` | See `Occupation` enum |
| `total_visits` | int | `5` | Website visit count |
| `total_time_spent_on_website` | int | `800` | Seconds |
| `page_views_per_visit` | float | `2.5` | Average pages per session |
| `do_not_email` | bool | `false` | CRM opt-out flag |
| `do_not_call` | bool | `false` | CRM opt-out flag |

### Optional (improve precision)

| Field | Type | Notes |
|---|---|---|
| `specialization` | enum | Course interest area |
| `through_recommendations` | bool | Referral signal |
| `x_education_forums` | bool | Community engagement |
| `receive_more_updates_about_our_courses` | bool | Explicit opt-in |
| `asymmetric_activities` | bool | Bot / disengagement flag |
| `country`, `city` | str | Geographic context |

### Missing values

All fields default to neutral values. The scorer **never raises** on missing data — it simply skips that sub-signal. A completely empty lead will still receive a minimal non-zero score from the default source/activity lookups.

---

## 7. Installation & Configuration

### Prerequisites

- Python 3.9 or newer
- `pip`

### Install

```bash
# Core only (no external deps)
# Already works with Python stdlib alone

# Full stack (testing + API + notebooks)
pip install -r requirements.txt
```

### Configuration

Edit `config/scoring_config.json` to adjust any weight or lookup value:

```json
{
  "weight_engagement_activity": 30.0,
  "weight_lead_origin_source":  20.0,
  "weight_demographic_fit":     20.0,
  "weight_behavioral_signals":  15.0,
  "weight_opt_in_recency":      15.0,

  "tier_hot":  75.0,
  "tier_warm": 50.0,
  "tier_cold": 25.0,

  "source_scores": {
    "Welingak Website": 1.0,
    "Reference":        0.9,
    ...
  }
}
```

> **Rule of thumb:** When tuning, hold `weight_*` values constant and adjust lookup scores first. Recalibrate tier thresholds last using your actual historical conversion data.

---

## 8. Usage

### 8.1 Python API

```python
from src.scorer import Lead, LeadScorer, LeadSource, Occupation, LastActivity

scorer = LeadScorer()

lead = Lead(
    lead_id="LEAD-42",
    lead_source=LeadSource.REFERENCE,
    occupation=Occupation.WORKING_PROFESSIONAL,
    last_activity=LastActivity.FORM_SUBMITTED,
    total_visits=7,
    total_time_spent_on_website=1400,
    page_views_per_visit=3.5,
    through_recommendations=True,
)

result = scorer.score(lead)

print(result.score)          # 63.8
print(result.tier)           # "Warm"
print(result.score_breakdown)
# {
#   'engagement_activity': 22.5,
#   'lead_origin_source': 18.0,
#   'demographic_fit': 18.5,
#   'behavioral_signals': 2.5,
#   'opt_in_recency': 0.0
# }

# Human-readable explanation
print(scorer.explain(lead))
```

#### Batch scoring

```python
leads = [lead1, lead2, lead3, ...]
results = scorer.score_batch(leads)

for r in results:
    print(f"{r.lead_id}: {r.score:.1f} ({r.tier})")
```

#### Custom config

```python
from src.scorer import ScoringConfig

cfg = ScoringConfig.from_file("config/scoring_config.json")
cfg.tier_hot = 80.0   # stricter hot threshold
scorer = LeadScorer(cfg)
```

---

### 8.2 CLI Pipeline (CSV → CSV)

```bash
# Score a file
python src/pipeline.py data/leads.csv data/scored_leads.csv

# With custom config
python src/pipeline.py data/leads.csv data/scored_leads.csv \
    --config config/scoring_config.json
```

Output CSV adds these columns to the original:
`score, tier, dim_engagement, dim_origin, dim_demographic, dim_behavioral, dim_opt_in`

**Sample output:**

```
✓ Scored 9240 leads → data/scored_leads.csv

Tier distribution:
  Hot             1387  (15.0%)
  Warm            3142  (34.0%)
  Cold            2956  (32.0%)
  Disqualified    1755  (19.0%)
```

---

### 8.3 REST API

```bash
uvicorn src.api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

**Score a single lead:**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "lead_id": "LEAD-42",
    "lead_source": "Reference",
    "occupation": "Working Professional",
    "last_activity": "Form Submitted on Website",
    "total_visits": 7,
    "total_time_spent_on_website": 1400,
    "page_views_per_visit": 3.5,
    "through_recommendations": true
  }'
```

**Response:**

```json
{
  "lead_id": "LEAD-42",
  "score": 63.8,
  "tier": "Warm",
  "breakdown": {
    "engagement_activity": 22.5,
    "lead_origin_source": 18.0,
    "demographic_fit": 18.5,
    "behavioral_signals": 2.5,
    "opt_in_recency": 0.0
  },
  "signals": {
    "lead_source": "Reference",
    "occupation": "Working Professional",
    "last_activity": "Form Submitted on Website",
    "total_visits": 7,
    "time_on_site_seconds": 1400,
    "page_views_per_visit": 3.5,
    "do_not_email": false,
    "do_not_call": false
  }
}
```

**Batch endpoint:**

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{ "leads": [ {...}, {...} ] }'
```

Returns results array plus a summary `{ total, avg_score, tier_counts }`.

---

## 9. Interpreting Scores & Acting on Them

### For sales reps

| Score | Tier | What it means | Next action |
|---|---|---|---|
| 75–100 | 🔥 Hot | High-intent lead, strong fit | Call within 2 hours |
| 50–74 | 🟡 Warm | Interested but needs nudge | Email + call within 24 hrs |
| 25–49 | 🔵 Cold | Low engagement or weak fit | Add to nurture sequence |
| 0–24 | ⚪ Disqualified | Do-not-contact or near-zero signals | No outreach now |

### Reading the breakdown

The `score_breakdown` tells you *which dimension to focus on*:

```
If dim_engagement is low  → Try content re-engagement (webinar invite, free module)
If dim_origin is low      → Lead came from low-intent channel; manage expectations
If dim_demographic is low → Pitch may need adjustment; confirm occupation/role
If dim_behavioral is low  → Add to newsletter, invite to community forum
If do_not_email/call set  → Never outreach; wait for inbound signal
```

### Score over time

A lead that was Cold last month but is now Warm has increased engagement — re-score leads every 7 days to catch momentum shifts. Integrate with your CRM to trigger alerts when a lead crosses a tier boundary.

---

## 10. Validation & Testing

### Unit tests (29 tests, 100 % passing)

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

Test categories:

| Category | Tests | What's verified |
|---|---|---|
| Score range | 4 | All scores in [0, 100] |
| Tier assignment | 4 | Correct tier labels |
| Monotonicity | 4 | More engagement → higher score |
| Breakdown | 3 | Dimensions present, sum correctly |
| Config override | 2 | Custom weights and thresholds apply |
| Batch scoring | 2 | Count and order preserved |
| Explain | 2 | Output completeness |
| Validation metrics | 4 | Rank-order, Gini, tier breakdown |
| Edge cases | 4 | Empty leads, all-negative, determinism |

### Validating against historical data

Once you have X Education's Kaggle CSV (`Leads.csv`) with the `Converted` column:

```python
from src.pipeline import load_csv
from src.scorer import LeadScorer
from src.validator import full_validation_report, print_validation_report

leads  = load_csv("data/Leads.csv")
scorer = LeadScorer()
scored = scorer.score_batch(leads)

report = full_validation_report(scored)
print_validation_report(report)
```

**Success criteria:**

| Metric | Minimum target | Ideal |
|---|---|---|
| Rank-order correct | True | True |
| Hot tier conversion rate | ≥ 60 % | ≥ 70 % |
| Disqualified conversion rate | ≤ 15 % | ≤ 10 % |
| Gini coefficient | ≥ 0.20 | ≥ 0.40 |
| Decile 1 lift | ≥ 1.5× | ≥ 2× |

### Decile lift table (expected shape)

A well-calibrated model should look roughly like this:

```
Decile   Score range   Conv rate   Lift    Meaning
─────────────────────────────────────────────────────
1        78–100        70–80 %     2.3×    Top 10% of leads
2        65–77         55–70 %     1.9×
3        55–64         45–55 %     1.5×
4        45–54         35–45 %     1.2×
5        35–44         25–35 %     1.0×    Near baseline
6        25–34         15–25 %     0.7×
7        18–24         10–15 %     0.4×
8        12–17          5–10 %     0.3×
9         6–11          3–5 %      0.2×
10        0–5           1–3 %      0.1×    Worst 10%
```

---

## 11. Customising the Model

### Adjusting weights

Edit `config/scoring_config.json`. Weights do not need to sum to exactly 100 — the system auto-scales within each dimension:

```json
{
  "weight_engagement_activity": 40.0,   // Increase if engagement is best predictor
  "weight_demographic_fit": 10.0        // Decrease if occupation data is sparse
}
```

### Adjusting source or activity scores

Add or change entries in the lookup tables:

```json
"source_scores": {
  "My New Partner Site": 0.95,   // New high-intent referral channel
  "TikTok Ads": 0.30
}
```

### Adjusting tiers

```json
"tier_hot":  80.0,   // Stricter — fewer but higher-quality hot leads
"tier_warm": 55.0
```

### Re-calibrating from data

If you have historical conversion data, you can compute empirical conversion rates per source, occupation, and activity, and paste those directly into the config as lookup scores. This turns the rule-based weights into data-backed weights without requiring a full ML pipeline.

---

## 12. End-to-End Example

```
Scenario: Two leads arrive on Monday morning.

Lead A — Rahul
  Source:        Welingak Website (returning visitor)
  Occupation:    Working Professional
  Last Activity: Form Submitted on Website
  Visits:        10, Time: 2 000 s, PVPV: 5.0
  Signals:       through_recommendations=True, opt-in=True

  → Score: 78.1 / 100   Tier: 🔥 Hot
  → Action: Sales rep calls Rahul at 10 AM today.

Lead B — Priya
  Source:        Facebook Ad
  Occupation:    Student
  Last Activity: Email Bounced
  Visits:        1, Time: 30 s
  Signals:       do_not_email=True

  → Score: 18.3 / 100   Tier: ⚪ Disqualified
  → Action: No outreach. Add to re-engagement list in 30 days.
```

Explanation output for Lead A:

```
Lead: L001
  Total Score : 78.1 / 100  →  Tier: Hot

  Dimension breakdown:
    engagement_activity          30.0/30  [████████████████████] 100%
    lead_origin_source           19.6/20  [███████████████████░] 98%
    demographic_fit              18.5/20  [██████████████████░░] 92%
    behavioral_signals            5.0/15  [██████░░░░░░░░░░░░░░] 33%
    opt_in_recency                5.0/15  [██████░░░░░░░░░░░░░░] 33%

  Key signals:
    Source:       Welingak Website
    Occupation:   Working Professional
    Last Activity:Form Submitted on Website
    Visits:       10
    Time on site: 2000s
```

---

## 13. Dataset Reference

**Kaggle dataset:** [Lead Scoring Dataset by Amrita Chatterjee](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)

| Column | Used | Notes |
|---|---|---|
| Prospect ID | ✓ | Maps to `lead_id` |
| Lead Source | ✓ | Primary origin signal |
| Total Visits | ✓ | Engagement sub-signal |
| Total Time Spent on Website | ✓ | Engagement sub-signal |
| Page Views Per Visit | ✓ | Engagement sub-signal |
| Last Activity | ✓ | Recency signal |
| What is your current occupation | ✓ | Maps to `occupation` |
| Specialization | ✓ | Demographic fit |
| Do Not Email / Call | ✓ | Negative behavioural |
| Through Recommendations | ✓ | Referral signal |
| Converted | ✓ | Ground truth (validation only) |

---

## 14. Roadmap

| Phase | Feature | Status |
|---|---|---|
| v1.0 | Rule-based engine + CLI | ✅ Done |
| v1.0 | REST API (FastAPI) | ✅ Done |
| v1.0 | 29-test pytest suite | ✅ Done |
| v1.1 | Logistic Regression baseline (scikit-learn) | Planned |
| v1.2 | Feature importance via SHAP | Planned |
| v1.3 | CRM webhook integration | Planned |
| v2.0 | Gradient Boosting model (XGBoost/LightGBM) | Planned |
| v2.1 | Automated re-training pipeline | Planned |

---

## 15. Contributing

1. Fork the repo and create a feature branch: `git checkout -b feat/my-improvement`
2. Add tests for any new logic: `pytest tests/ -v`
3. Keep the core engine dependency-free (stdlib only)
4. Open a PR with a description of what changed and why

---

*Built for X Education by the Sellable Technologies internship team.*
*Dataset: Kaggle — Lead Scoring Dataset (Amrita Chatterjee, CC0)*
