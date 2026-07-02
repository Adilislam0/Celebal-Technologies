"""
lead_scoring/src/validator.py
Validates scoring model against historical conversion data.

Key metric: Lift — do higher-scoring leads actually convert at a higher rate?
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from scorer import Lead


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def conversion_rate_by_tier(scored_leads: List[Lead]) -> Dict[str, dict]:
    """Conversion rate per tier (requires lead.converted to be set)."""
    labelled = [l for l in scored_leads if l.converted is not None]
    if not labelled:
        return {}

    by_tier: Dict[str, List[bool]] = defaultdict(list)
    for l in labelled:
        by_tier[l.tier or "Unknown"].append(bool(l.converted))

    result = {}
    for tier in ["Hot", "Warm", "Cold", "Disqualified"]:
        flags = by_tier.get(tier, [])
        if not flags:
            result[tier] = {"count": 0, "converted": 0, "rate": None}
            continue
        converted = sum(flags)
        result[tier] = {
            "count": len(flags),
            "converted": converted,
            "rate": round(converted / len(flags), 4),
        }
    return result


def lift_by_decile(scored_leads: List[Lead]) -> List[dict]:
    """
    Rank leads by score, split into 10 deciles, report conversion rate per
    decile.  Ideal model: decile 1 (highest scores) has highest conversion.
    """
    labelled = [l for l in scored_leads if l.converted is not None and l.score is not None]
    if not labelled:
        return []

    labelled.sort(key=lambda l: l.score, reverse=True)
    n = len(labelled)
    overall_rate = sum(l.converted for l in labelled) / n

    deciles = []
    size = math.ceil(n / 10)
    for i in range(10):
        chunk = labelled[i * size: (i + 1) * size]
        if not chunk:
            break
        conv = sum(l.converted for l in chunk)
        rate = conv / len(chunk)
        lift = rate / overall_rate if overall_rate > 0 else 0
        deciles.append({
            "decile": i + 1,
            "score_range": f"{chunk[-1].score:.0f}–{chunk[0].score:.0f}",
            "count": len(chunk),
            "converted": conv,
            "conversion_rate": round(rate, 4),
            "lift": round(lift, 2),
        })
    return deciles


def rank_order_test(scored_leads: List[Lead]) -> bool:
    """
    Sanity check: avg score of converted leads > avg score of non-converted.
    Returns True if the model is directionally correct.
    """
    labelled = [l for l in scored_leads if l.converted is not None and l.score is not None]
    conv   = [l.score for l in labelled if l.converted]
    noconv = [l.score for l in labelled if not l.converted]
    if not conv or not noconv:
        return True
    return (sum(conv) / len(conv)) > (sum(noconv) / len(noconv))


def gini_coefficient(scored_leads: List[Lead]) -> float:
    """
    Normalised Gini coefficient as a single model quality number.
    Perfect model = 1.0; random model = 0.0.
    Computed from the cumulative gains curve.
    """
    labelled = [l for l in scored_leads if l.converted is not None and l.score is not None]
    labelled.sort(key=lambda l: l.score, reverse=True)
    n = len(labelled)
    total_conv = sum(l.converted for l in labelled)
    if total_conv == 0 or n == 0:
        return 0.0

    cumulative_gains = 0.0
    for i, lead in enumerate(labelled):
        cumulative_gains += (lead.converted / total_conv) - (1 / n)
    gini = 2 * cumulative_gains / n
    return round(gini, 4)


def full_validation_report(scored_leads: List[Lead]) -> dict:
    """Aggregate all validation metrics into a single dict."""
    return {
        "total_leads":           len(scored_leads),
        "labelled_leads":        sum(1 for l in scored_leads if l.converted is not None),
        "overall_conversion_rate": _overall_rate(scored_leads),
        "rank_order_correct":    rank_order_test(scored_leads),
        "gini_coefficient":      gini_coefficient(scored_leads),
        "tier_breakdown":        conversion_rate_by_tier(scored_leads),
        "decile_lift":           lift_by_decile(scored_leads),
    }


def print_validation_report(report: dict) -> None:
    print("=" * 60)
    print("LEAD SCORING VALIDATION REPORT")
    print("=" * 60)
    print(f"Total leads:          {report['total_leads']}")
    print(f"Labelled leads:       {report['labelled_leads']}")
    print(f"Overall conv. rate:   {report['overall_conversion_rate']:.1%}")
    print(f"Rank-order correct:   {'✓ Yes' if report['rank_order_correct'] else '✗ No'}")
    print(f"Gini coefficient:     {report['gini_coefficient']:.4f}")
    print()
    print("Conversion rate by tier:")
    for tier, stats in report["tier_breakdown"].items():
        if stats["rate"] is None:
            print(f"  {tier:<15} (no data)")
            continue
        bar = "▓" * int(stats["rate"] * 20)
        print(f"  {tier:<15} {stats['rate']:5.1%}  {bar}  (n={stats['count']})")
    print()
    print("Decile lift analysis:")
    print(f"  {'Decile':<8} {'Score range':<14} {'Conv rate':<12} {'Lift':<8} {'Count'}")
    print(f"  {'-'*55}")
    for d in report["decile_lift"]:
        flag = "🔥" if d["lift"] >= 2 else ("✓" if d["lift"] >= 1 else "↓")
        print(f"  {d['decile']:<8} {d['score_range']:<14} {d['conversion_rate']:<12.1%} {d['lift']:<8.2f} {d['count']} {flag}")
    print("=" * 60)


def _overall_rate(leads: List[Lead]) -> float:
    labelled = [l for l in leads if l.converted is not None]
    if not labelled:
        return 0.0
    return sum(l.converted for l in labelled) / len(labelled)
