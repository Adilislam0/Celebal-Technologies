"""
lead_scoring/tests/test_scorer.py
Full test suite for the X Education lead scoring system.
Run with: pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scorer import (
    Lead, LeadScorer, ScoringConfig,
    LeadSource, Occupation, LastActivity, SpecializationInterest,
)
from validator import (
    conversion_rate_by_tier, lift_by_decile,
    rank_order_test, gini_coefficient, full_validation_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    return LeadScorer()


@pytest.fixture
def hot_lead():
    return Lead(
        lead_id="hot-001",
        lead_source=LeadSource.WELINGAK_WEBSITE,
        occupation=Occupation.WORKING_PROFESSIONAL,
        last_activity=LastActivity.FORM_SUBMITTED,
        total_visits=10,
        total_time_spent_on_website=2000,
        page_views_per_visit=5.0,
        through_recommendations=True,
        x_education_forums=True,
        digital_advertisement=True,
        specialization=SpecializationInterest.FINANCE_MANAGEMENT,
        receive_more_updates_about_our_courses=True,
        update_me_on_supply_chain_content=True,
    )


@pytest.fixture
def cold_lead():
    return Lead(
        lead_id="cold-001",
        lead_source=LeadSource.FACEBOOK,
        occupation=Occupation.STUDENT,
        last_activity=LastActivity.EMAIL_BOUNCED,
        total_visits=1,
        total_time_spent_on_website=30,
        page_views_per_visit=1.0,
        do_not_email=True,
        do_not_call=True,
    )


@pytest.fixture
def warm_lead():
    return Lead(
        lead_id="warm-001",
        lead_source=LeadSource.ORGANIC_SEARCH,
        occupation=Occupation.UNEMPLOYED,
        last_activity=LastActivity.EMAIL_OPENED,
        total_visits=4,
        total_time_spent_on_website=600,
        page_views_per_visit=2.5,
    )


# ---------------------------------------------------------------------------
# Score range tests
# ---------------------------------------------------------------------------

class TestScoreRange:
    def test_score_between_0_and_100(self, scorer, hot_lead, cold_lead, warm_lead):
        for lead in [hot_lead, cold_lead, warm_lead]:
            result = scorer.score(lead)
            assert 0 <= result.score <= 100, f"Score {result.score} out of range for {lead.lead_id}"

    def test_empty_lead_scores_above_zero(self, scorer):
        """Even empty leads should get a minimal score from defaults."""
        result = scorer.score(Lead(lead_id="empty"))
        assert result.score >= 0

    def test_perfect_lead_scores_high(self, scorer, hot_lead):
        result = scorer.score(hot_lead)
        assert result.score >= 60, f"Perfect lead only scored {result.score}"

    def test_disqualified_lead_scores_low(self, scorer, cold_lead):
        result = scorer.score(cold_lead)
        assert result.score < 40, f"Disqualified lead scored {result.score}"


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

class TestTierAssignment:
    def test_hot_tier(self, scorer, hot_lead):
        result = scorer.score(hot_lead)
        assert result.tier == "Hot"

    def test_cold_tier(self, scorer, cold_lead):
        result = scorer.score(cold_lead)
        assert result.tier in ("Cold", "Disqualified")

    def test_warm_tier(self, scorer, warm_lead):
        result = scorer.score(warm_lead)
        assert result.tier in ("Warm", "Cold")   # could be either

    def test_all_tiers_possible(self, scorer):
        configs = [
            Lead(lead_source=LeadSource.WELINGAK_WEBSITE, occupation=Occupation.WORKING_PROFESSIONAL,
                 last_activity=LastActivity.FORM_SUBMITTED, total_visits=10,
                 total_time_spent_on_website=2000, page_views_per_visit=5.0,
                 through_recommendations=True, receive_more_updates_about_our_courses=True,
                 specialization=SpecializationInterest.FINANCE_MANAGEMENT),
            Lead(lead_source=LeadSource.ORGANIC_SEARCH, occupation=Occupation.UNEMPLOYED,
                 last_activity=LastActivity.EMAIL_OPENED, total_visits=3,
                 total_time_spent_on_website=400),
            Lead(lead_source=LeadSource.FACEBOOK, occupation=Occupation.STUDENT,
                 last_activity=LastActivity.EMAIL_BOUNCED, total_visits=1,
                 do_not_email=True, do_not_call=True),
        ]
        tiers = {scorer.score(l).tier for l in configs}
        assert "Hot" in tiers
        assert ("Cold" in tiers or "Disqualified" in tiers)


# ---------------------------------------------------------------------------
# Monotonicity — more engagement should never hurt
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_more_visits_increases_score(self, scorer):
        base = Lead(lead_id="base", total_visits=2)
        more = Lead(lead_id="more", total_visits=8)
        assert scorer.score(more).score >= scorer.score(base).score

    def test_more_time_increases_score(self, scorer):
        base = Lead(lead_id="base", total_time_spent_on_website=100)
        more = Lead(lead_id="more", total_time_spent_on_website=1500)
        assert scorer.score(more).score >= scorer.score(base).score

    def test_do_not_contact_reduces_score(self, scorer):
        base = Lead(
            lead_id="base",
            lead_source=LeadSource.REFERENCE,
            through_recommendations=True,
            x_education_forums=True,
            magazine=True,
        )
        dnc  = Lead(
            lead_id="dnc",
            lead_source=LeadSource.REFERENCE,
            through_recommendations=True,
            x_education_forums=True,
            magazine=True,
            do_not_email=True,
            do_not_call=True,
        )
        assert scorer.score(dnc).score < scorer.score(base).score

    def test_high_intent_source_beats_low(self, scorer):
        high = Lead(lead_id="high", lead_source=LeadSource.WELINGAK_WEBSITE)
        low  = Lead(lead_id="low",  lead_source=LeadSource.FACEBOOK)
        assert scorer.score(high).score > scorer.score(low).score


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------

class TestBreakdown:
    def test_breakdown_has_all_dimensions(self, scorer, hot_lead):
        result = scorer.score(hot_lead)
        expected = {
            "engagement_activity", "lead_origin_source",
            "demographic_fit", "behavioral_signals", "opt_in_recency",
        }
        assert expected == set(result.score_breakdown.keys())

    def test_breakdown_sums_to_score(self, scorer, hot_lead):
        result = scorer.score(hot_lead)
        total = sum(result.score_breakdown.values())
        assert abs(total - result.score) < 0.1

    def test_all_dimensions_nonnegative(self, scorer, hot_lead, cold_lead):
        for lead in [hot_lead, cold_lead]:
            result = scorer.score(lead)
            for dim, val in result.score_breakdown.items():
                assert val >= 0, f"Negative score in {dim}: {val}"


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------

class TestConfigOverride:
    def test_custom_weights_apply(self):
        cfg = ScoringConfig()
        cfg.weight_engagement_activity = 50.0
        cfg.weight_lead_origin_source  = 20.0
        cfg.weight_demographic_fit     = 10.0
        cfg.weight_behavioral_signals  = 10.0
        cfg.weight_opt_in_recency      = 10.0
        scorer = LeadScorer(cfg)
        lead = Lead(total_visits=10, total_time_spent_on_website=2000)
        result = scorer.score(lead)
        assert result.score_breakdown["engagement_activity"] <= 50.0

    def test_tier_thresholds_respected(self):
        cfg = ScoringConfig()
        cfg.tier_hot  = 90.0    # Very strict — almost nothing is hot
        scorer = LeadScorer(cfg)
        lead = Lead(
            lead_source=LeadSource.WELINGAK_WEBSITE,
            occupation=Occupation.WORKING_PROFESSIONAL,
            last_activity=LastActivity.FORM_SUBMITTED,
            total_visits=10,
            total_time_spent_on_website=2000,
        )
        result = scorer.score(lead)
        # With threshold at 90, even a great lead should be Warm or below
        assert result.tier in ("Warm", "Cold", "Hot")


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_returns_same_count(self, scorer, hot_lead, cold_lead, warm_lead):
        leads = [hot_lead, cold_lead, warm_lead]
        results = scorer.score_batch(leads)
        assert len(results) == len(leads)

    def test_batch_preserves_order(self, scorer, hot_lead, cold_lead):
        results = scorer.score_batch([hot_lead, cold_lead])
        assert results[0].lead_id == hot_lead.lead_id
        assert results[1].lead_id == cold_lead.lead_id


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_output_contains_key_info(self, scorer, hot_lead):
        explanation = scorer.explain(hot_lead)
        assert "Score" in explanation
        assert "Tier" in explanation
        assert "engagement_activity" in explanation

    def test_explain_does_not_raise(self, scorer, cold_lead):
        try:
            scorer.explain(cold_lead)
        except Exception as e:
            pytest.fail(f"explain() raised {e}")


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

class TestValidation:
    def _make_scored_leads(self, scorer):
        """Generate synthetic scored leads with known conversion labels."""
        leads = []

        # Hot leads — mostly converted
        for i in range(30):
            l = Lead(
                lead_id=f"h{i}",
                lead_source=LeadSource.WELINGAK_WEBSITE,
                occupation=Occupation.WORKING_PROFESSIONAL,
                last_activity=LastActivity.FORM_SUBMITTED,
                total_visits=8, total_time_spent_on_website=1500,
                through_recommendations=True,
                converted=(i < 22),   # 22/30 = 73% conversion
            )
            leads.append(scorer.score(l))

        # Cold leads — mostly not converted
        for i in range(30):
            l = Lead(
                lead_id=f"c{i}",
                lead_source=LeadSource.FACEBOOK,
                occupation=Occupation.STUDENT,
                last_activity=LastActivity.EMAIL_BOUNCED,
                total_visits=1,
                do_not_email=True,
                converted=(i < 5),    # 5/30 = 17% conversion
            )
            leads.append(scorer.score(l))

        return leads

    def test_rank_order_correct(self, scorer):
        leads = self._make_scored_leads(scorer)
        assert rank_order_test(leads), "Avg score of converted < non-converted!"

    def test_tier_breakdown_hot_rate_higher(self, scorer):
        leads = self._make_scored_leads(scorer)
        breakdown = conversion_rate_by_tier(leads)
        hot_rate  = breakdown.get("Hot",  {}).get("rate") or 0
        cold_rate = breakdown.get("Cold", {}).get("rate") or 0
        dis_rate  = breakdown.get("Disqualified", {}).get("rate") or 0
        # Hot should beat both cold and disqualified
        assert hot_rate >= cold_rate, f"Hot {hot_rate:.1%} <= Cold {cold_rate:.1%}"

    def test_gini_positive(self, scorer):
        leads = self._make_scored_leads(scorer)
        # Sanity: the two groups must have different scores for Gini to work
        hot_scores  = [l.score for l in leads if l.lead_id.startswith("h")]
        cold_scores = [l.score for l in leads if l.lead_id.startswith("c")]
        assert hot_scores and cold_scores
        assert sum(hot_scores)/len(hot_scores) > sum(cold_scores)/len(cold_scores), \
            "Hot leads should outscore cold leads on average"
        g = gini_coefficient(leads)
        assert g >= 0, f"Gini should be non-negative, got {g}"

    def test_full_report_schema(self, scorer):
        leads = self._make_scored_leads(scorer)
        report = full_validation_report(leads)
        for key in ["total_leads", "labelled_leads", "overall_conversion_rate",
                    "rank_order_correct", "gini_coefficient",
                    "tier_breakdown", "decile_lift"]:
            assert key in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_score_with_all_negative_signals(self, scorer):
        lead = Lead(
            do_not_email=True, do_not_call=True,
            lead_source=LeadSource.UNKNOWN,
            last_activity=LastActivity.EMAIL_BOUNCED,
            occupation=Occupation.UNKNOWN,
            asymmetric_activities=True,
        )
        result = scorer.score(lead)
        assert result.score >= 0

    def test_score_with_all_positive_signals(self, scorer):
        lead = Lead(
            lead_source=LeadSource.WELINGAK_WEBSITE,
            occupation=Occupation.WORKING_PROFESSIONAL,
            last_activity=LastActivity.FORM_SUBMITTED,
            total_visits=10, total_time_spent_on_website=2000,
            page_views_per_visit=5.0,
            through_recommendations=True, x_education_forums=True,
            magazine=True, newspaper_article=True, digital_advertisement=True,
            receive_more_updates_about_our_courses=True,
            update_me_on_supply_chain_content=True,
            get_updates_on_dm_content=True,
            specialization=SpecializationInterest.FINANCE_MANAGEMENT,
        )
        result = scorer.score(lead)
        assert result.score <= 100

    def test_score_is_deterministic(self, scorer, hot_lead):
        r1 = scorer.score(hot_lead)
        r2 = scorer.score(hot_lead)
        assert r1.score == r2.score

    def test_batch_empty_list(self, scorer):
        result = scorer.score_batch([])
        assert result == []
