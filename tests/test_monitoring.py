"""Тесты для MonitoringAgent (без Mesa — standalone)."""
import numpy as np
import pytest

# Тестируем логику валидации напрямую, без инициализации Mesa
from credit_scoring_phe.agents.monitoring_agent import MonitoringAgent

NUMERIC_BOUNDS    = MonitoringAgent.NUMERIC_BOUNDS
CATEGORICAL_VALUES = MonitoringAgent.CATEGORICAL_VALUES


def _validate(row: dict):
    """Standalone-вызов логики валидации без Mesa."""
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in row:
            continue
        val = row[col]
        if not isinstance(val, (int, float, np.integer, np.floating)):
            return False, f"bad type: {col}"
        if not (lo <= val <= hi):
            return False, f"out of range: {col}={val}"
    for col, allowed in CATEGORICAL_VALUES.items():
        if col not in row:
            continue
        if str(row[col]) not in allowed:
            return False, f"invalid categorical: {col}={row[col]}"
    for col in NUMERIC_BOUNDS:
        if col in row and (
            row[col] is None
            or (isinstance(row[col], float) and np.isnan(row[col]))
        ):
            return False, f"NaN: {col}"
    return True, None


VALID_ROW = {
    "Age": 35, "Sex": "male", "Job": 2, "Housing": "own",
    "Saving accounts": "little", "Checking account": "none",
    "Credit amount": 3000, "Duration": 12, "Purpose": "car",
}


class TestValidRow:
    def test_valid_row_passes(self):
        valid, err = _validate(VALID_ROW)
        assert valid is True
        assert err is None

    def test_boundary_min_passes(self):
        row = {**VALID_ROW, "Age": 18, "Job": 0,
               "Credit amount": 100, "Duration": 1}
        valid, _ = _validate(row)
        assert valid is True

    def test_boundary_max_passes(self):
        row = {**VALID_ROW, "Age": 100, "Job": 3,
               "Credit amount": 200_000, "Duration": 120}
        valid, _ = _validate(row)
        assert valid is True


class TestNumericBounds:
    def test_negative_age_blocked(self):
        valid, err = _validate({**VALID_ROW, "Age": -5})
        assert valid is False
        assert "Age" in err

    def test_age_over_100_blocked(self):
        valid, err = _validate({**VALID_ROW, "Age": 101})
        assert valid is False

    def test_credit_over_200k_blocked(self):
        valid, err = _validate({**VALID_ROW, "Credit amount": 200_001})
        assert valid is False

    def test_duration_zero_blocked(self):
        valid, err = _validate({**VALID_ROW, "Duration": 0})
        assert valid is False

    def test_job_negative_blocked(self):
        valid, err = _validate({**VALID_ROW, "Job": -1})
        assert valid is False

    def test_duration_24_passes(self):
        """Duration=24 — граница, должна пропускаться (условие > 24)."""
        valid, _ = _validate({**VALID_ROW, "Duration": 24})
        assert valid is True

    def test_credit_5000_passes(self):
        """Credit amount=5000 — граница, должна пропускаться (условие > 5000)."""
        valid, _ = _validate({**VALID_ROW, "Credit amount": 5000})
        assert valid is True


class TestCategorical:
    def test_invalid_sex_blocked(self):
        valid, err = _validate({**VALID_ROW, "Sex": "unknown"})
        assert valid is False
        assert "Sex" in err

    def test_invalid_housing_blocked(self):
        valid, err = _validate({**VALID_ROW, "Housing": "hotel"})
        assert valid is False

    def test_all_valid_sex_values(self):
        for val in ("male", "female"):
            valid, _ = _validate({**VALID_ROW, "Sex": val})
            assert valid is True, f"Expected {val} to pass"

    def test_all_valid_housing_values(self):
        for val in ("own", "free", "rent"):
            valid, _ = _validate({**VALID_ROW, "Housing": val})
            assert valid is True


class TestNaN:
    def test_nan_age_blocked(self):
        valid, err = _validate({**VALID_ROW, "Age": float("nan")})
        assert valid is False

    def test_none_age_blocked(self):
        valid, err = _validate({**VALID_ROW, "Age": None})
        assert valid is False
