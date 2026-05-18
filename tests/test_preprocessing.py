"""Тесты предобработки: риск-скор и encode_and_split."""
import numpy as np
import pandas as pd
import pytest

from credit_scoring_phe.data.preprocessing import compute_risk_score, add_target


def _make_row(saving="little", checking="little", housing="rent",
              duration=36, amount=10_000):
    return pd.Series({
        "Saving accounts": saving, "Checking account": checking,
        "Housing": housing, "Duration": duration, "Credit amount": amount,
    })


class TestComputeRiskScore:
    def test_best_profile(self):
        row = _make_row("rich", "rich", "own", 12, 3000)
        assert compute_risk_score(row) == 2 + 2 + 1 + 1 + 1

    def test_worst_profile(self):
        row = _make_row("little", "little", "rent", 36, 10_000)
        assert compute_risk_score(row) == -1 - 1 - 1 - 1 - 1

    def test_duration_boundary(self):
        """Duration=24 → не > 24 → +1."""
        row = _make_row("none", "none", "free", 24, 1000)
        score = compute_risk_score(row)
        assert score == -1 - 1 + 0 + 1 + 1

    def test_duration_over_boundary(self):
        """Duration=25 → > 24 → -1."""
        row = _make_row("none", "none", "free", 25, 1000)
        score = compute_risk_score(row)
        assert score == -1 - 1 + 0 - 1 + 1

    def test_amount_boundary(self):
        """Credit amount=5000 → не > 5000 → +1."""
        row = _make_row("none", "none", "free", 12, 5000)
        score = compute_risk_score(row)
        assert score == -1 - 1 + 0 + 1 + 1

    def test_nan_saving_defaults_minus1(self):
        """NaN/неизвестное значение сбережений → default -1."""
        row = _make_row(saving=None, checking="moderate", housing="own",
                        duration=12, amount=1000)
        score = compute_risk_score(row)
        assert score == -1 + 1 + 1 + 1 + 1

    def test_target_distribution(self):
        """Good credit ratio должен быть ~70% при 30-м перцентиле."""
        import pandas as pd
        rows = []
        for s in ["little", "moderate", "rich", "quite rich", None]:
            for c in ["little", "moderate", "rich", None]:
                for h in ["own", "free", "rent"]:
                    for d in [12, 24, 36]:
                        for a in [3000, 5000, 10000]:
                            rows.append({
                                "Saving accounts": s, "Checking account": c,
                                "Housing": h, "Duration": d, "Credit amount": a,
                                "Age": 30, "Sex": "male", "Job": 2,
                                "Purpose": "car",
                            })
        df = pd.DataFrame(rows)
        df = add_target(df)
        ratio = df["target"].mean()
        assert 0.60 <= ratio <= 0.85, f"Unexpected ratio: {ratio:.1%}"


class TestNoDataLeakage:
    """Проверяем, что scaler обучен только на train."""
    def test_scaler_fitted_on_train_only(self):
        from credit_scoring_phe.data.preprocessing import encode_and_split

        rows = []
        for i in range(200):
            rows.append({
                "Age": 20 + i % 60, "Sex": "male", "Job": i % 4,
                "Housing": "own", "Saving accounts": "little",
                "Checking account": "none", "Credit amount": 1000 + i * 10,
                "Duration": 12, "Purpose": "car", "target": i % 2,
            })
        df = pd.DataFrame(rows)
        X_train, X_test, y_train, y_test, _, scaler = encode_and_split(df)

        # Scaler должен быть обучен на X_train (160 строк), а не на 200
        assert scaler.n_samples_seen_ == len(X_train)
