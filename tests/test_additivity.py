"""Tests for additivity defect computation."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.additivity import compute_additivity_defect, compute_full_additivity_defect


def make_mock_llc(mean: float, std: float = 0.1):
    """Create a mock LLC result dict."""
    return {"llc_mean": mean, "llc_std": std, "llc_per_chain": [mean] * 5}


class TestComputeAdditivityDefect:
    def test_zero_defect(self):
        """When LLC is exactly additive, defect should be ~0."""
        results = {
            "{0}": make_mock_llc(2.0),
            "{1}": make_mock_llc(3.0),
            "{0}∪{1}": make_mock_llc(5.0),  # 2 + 3 = 5, perfect additivity
        }
        df = compute_additivity_defect(results, [("{0}", "{1}", "{0}∪{1}")])
        assert len(df) == 1
        assert abs(df.iloc[0]["delta"]) < 1e-10

    def test_positive_defect(self):
        """Positive defect means joint LLC < sum of parts (circuit sharing)."""
        results = {
            "{0}": make_mock_llc(2.0),
            "{1}": make_mock_llc(3.0),
            "{0}∪{1}": make_mock_llc(4.0),  # less than 2+3=5
        }
        df = compute_additivity_defect(results, [("{0}", "{1}", "{0}∪{1}")])
        assert df.iloc[0]["delta"] == pytest.approx(1.0)

    def test_negative_defect(self):
        """Negative defect means joint LLC > sum of parts."""
        results = {
            "{0}": make_mock_llc(2.0),
            "{1}": make_mock_llc(3.0),
            "{0}∪{1}": make_mock_llc(6.0),  # more than 2+3=5
        }
        df = compute_additivity_defect(results, [("{0}", "{1}", "{0}∪{1}")])
        assert df.iloc[0]["delta"] == pytest.approx(-1.0)

    def test_error_propagation(self):
        """Standard error of defect should be sqrt(sum of variances)."""
        results = {
            "{0}": make_mock_llc(2.0, std=0.3),
            "{1}": make_mock_llc(3.0, std=0.4),
            "{0}∪{1}": make_mock_llc(5.0, std=0.5),
        }
        df = compute_additivity_defect(results, [("{0}", "{1}", "{0}∪{1}")])
        expected_std = math.sqrt(0.3**2 + 0.4**2 + 0.5**2)
        assert df.iloc[0]["delta_std"] == pytest.approx(expected_std)

    def test_zscore(self):
        results = {
            "{0}": make_mock_llc(2.0, std=0.1),
            "{1}": make_mock_llc(3.0, std=0.1),
            "{0}∪{1}": make_mock_llc(4.0, std=0.1),
        }
        df = compute_additivity_defect(results, [("{0}", "{1}", "{0}∪{1}")])
        delta = 1.0
        delta_std = math.sqrt(0.01 * 3)
        assert df.iloc[0]["delta_zscore"] == pytest.approx(delta / delta_std)

    def test_multiple_triples(self):
        results = {
            "{0}": make_mock_llc(2.0),
            "{1}": make_mock_llc(3.0),
            "{2}": make_mock_llc(2.5),
            "{0}∪{1}": make_mock_llc(5.0),
            "{0}∪{2}": make_mock_llc(4.0),
        }
        triples = [("{0}", "{1}", "{0}∪{1}"), ("{0}", "{2}", "{0}∪{2}")]
        df = compute_additivity_defect(results, triples)
        assert len(df) == 2
        assert df.iloc[0]["delta"] == pytest.approx(0.0)
        assert df.iloc[1]["delta"] == pytest.approx(0.5)

    def test_dataframe_columns(self):
        results = {
            "{0}": make_mock_llc(1.0),
            "{1}": make_mock_llc(1.0),
            "{0}∪{1}": make_mock_llc(2.0),
        }
        df = compute_additivity_defect(results, [("{0}", "{1}", "{0}∪{1}")])
        expected_cols = {"T1", "T2", "T_joint", "llc_T1", "llc_T2",
                         "llc_joint", "delta", "delta_std", "delta_zscore"}
        assert set(df.columns) == expected_cols


class TestComputeFullAdditivityDefect:
    def test_basic(self):
        results = {
            "{0}": make_mock_llc(1.0),
            "{1}": make_mock_llc(2.0),
            "{2}": make_mock_llc(1.5),
            "{3}": make_mock_llc(2.5),
            "composite": make_mock_llc(5.0),
        }
        r = compute_full_additivity_defect(
            results,
            atomic_names=["{0}", "{1}", "{2}", "{3}"],
            composite_name="composite",
        )
        # sum = 1+2+1.5+2.5 = 7, composite = 5, delta = 2
        assert r["sum_atomic_llc"] == pytest.approx(7.0)
        assert r["composite_llc"] == pytest.approx(5.0)
        assert r["delta"] == pytest.approx(2.0)

    def test_error_propagation(self):
        results = {
            "{0}": make_mock_llc(1.0, std=0.1),
            "{1}": make_mock_llc(1.0, std=0.2),
            "composite": make_mock_llc(2.0, std=0.3),
        }
        r = compute_full_additivity_defect(
            results, atomic_names=["{0}", "{1}"], composite_name="composite"
        )
        expected_std = math.sqrt(0.01 + 0.04 + 0.09)
        assert r["delta_std"] == pytest.approx(expected_std)
