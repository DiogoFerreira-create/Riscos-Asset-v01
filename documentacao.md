"""Testes de risk_models.returns: retornos log, winsorização e fonte sintética."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_models import DataSource, download_market_data, log_returns, winsorize_returns
from risk_models.exceptions import DataValidationError


class TestLogReturns:
    def test_known_value(self) -> None:
        """ln(110/100) deve sair exato — verificação de fórmula."""
        prices = pd.DataFrame({"X": [100.0, 110.0]})
        result = log_returns(prices)
        assert result["X"].iloc[0] == pytest.approx(np.log(1.1))

    def test_rejects_nonpositive_prices(self) -> None:
        prices = pd.DataFrame({"X": [100.0, 0.0, 90.0]})
        with pytest.raises(DataValidationError, match="não positivos"):
            log_returns(prices)

    def test_rejects_single_row(self) -> None:
        with pytest.raises(DataValidationError):
            log_returns(pd.DataFrame({"X": [100.0]}))


class TestWinsorize:
    def test_bounds_are_respected(self, returns_frame: pd.DataFrame) -> None:
        """Após winsorizar em [1%, 99%], nada pode exceder esses quantis."""
        wz = winsorize_returns(returns_frame, 0.01, 0.99)
        lo = returns_frame.quantile(0.01)
        hi = returns_frame.quantile(0.99)
        assert (wz.max() <= hi + 1e-12).all()
        assert (wz.min() >= lo - 1e-12).all()

    def test_interior_values_untouched(self, returns_frame: pd.DataFrame) -> None:
        """Mediana está longe das caudas: winsorizar não deve alterá-la."""
        wz = winsorize_returns(returns_frame, 0.01, 0.99)
        pd.testing.assert_series_equal(wz.median(), returns_frame.median())

    def test_invalid_percentiles(self, returns_frame: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            winsorize_returns(returns_frame, 0.99, 0.01)


class TestSyntheticSource:
    def test_reproducible_with_seed(self) -> None:
        """Mesma semente => mesmos preços (contrato de reprodutibilidade)."""
        a = download_market_data(("T1", "T2"), "2022-01-03", "2022-12-30",
                                 source=DataSource.SYNTHETIC, synthetic_seed=7)
        b = download_market_data(("T1", "T2"), "2022-01-03", "2022-12-30",
                                 source=DataSource.SYNTHETIC, synthetic_seed=7)
        pd.testing.assert_frame_equal(a.prices, b.prices)

    def test_positive_prices_and_volumes(self) -> None:
        md = download_market_data(("T1",), "2022-01-03", "2022-06-30",
                                  source=DataSource.SYNTHETIC)
        assert (md.prices > 0).all().all()
        assert (md.volumes > 0).all().all()
