"""
Pruebas unitarias para el motor de riesgo cuantitativo FX (FXRiskEngine).
Incluye verificación de convexidad, coherencia VaR/ES, Trigger Rate, TWAP 5d,
cobertura FX Forward y contrastes de Kupiec (POF) y Christoffersen (1998).
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from src.risk_engine import FXRiskEngine, PositionConfig


class TestFXRiskEngine:
    """Verificación cuantitativa de modelos de riesgo, valoraciones, benchmarks y backtesting."""

    @pytest.fixture
    def mock_engine(self):
        """Genera un motor de riesgo con cotizaciones sintéticas controladas."""
        np.random.seed(42)
        n = 650
        # Simular paseo aleatorio geométrico con S0 = 1.10
        returns = np.random.normal(0.0001, 0.005, n)
        spots = 1.10 * np.exp(np.cumsum(returns))
        dates = pd.date_range("2024-01-01", periods=n, freq="B").date

        df_synthetic = pd.DataFrame({
            "rate_date": dates,
            "spot_rate": spots,
            "ingested_at": [pd.Timestamp.now()] * n,
        })

        engine = FXRiskEngine(pos_config=PositionConfig(usd_amount=100_000.0, horizon_days=5, confidence_level=0.99))
        engine._df_rates = df_synthetic
        return engine

    def test_position_valuation_convexity(self, mock_engine):
        """Verifica la relación inversa y no lineal V = USD / S."""
        ctx = mock_engine.get_spot_context()
        S = ctx["spot_rate"]
        expected_v_eur = 100_000.0 / S
        assert np.isclose(ctx["current_v_eur"], expected_v_eur, atol=1e-4)

    def test_historical_var_es_and_trigger_rate(self, mock_engine):
        """
        Verifica:
        1. VaR(99%) > VaR(95%)
        2. Expected Shortfall(99%) >= VaR(99%)
        3. Exactitud algebraica del Trigger Rate: 100,000 / S_trigger == V_0 - VaR
        """
        ctx = mock_engine.get_spot_context()
        V0 = ctx["current_v_eur"]
        S0 = ctx["spot_rate"]

        res_99 = mock_engine.calculate_historical_var_es(confidence=0.99)
        res_95 = mock_engine.calculate_historical_var_es(confidence=0.95)

        assert res_99["var_eur"] > res_95["var_eur"], "El VaR al 99% debe ser superior al VaR al 95%"
        assert res_99["es_eur"] >= res_99["var_eur"], "Expected Shortfall (CVaR) debe ser mayor o igual que el VaR"
        assert res_99["worst_loss_eur"] >= res_99["es_eur"], "La peor pérdida histórica debe ser mayor o igual que el ES"
        assert res_99["trigger_spot_var"] > S0, "El Trigger Rate de pérdida debe implicar una subida de S (depreciación USD)"

        # Comprobar identidad exacta: 100,000 / trigger_spot_var == V0 - var_eur
        implied_val = 100_000.0 / res_99["trigger_spot_var"]
        assert np.isclose(implied_val, V0 - res_99["var_eur"], atol=1e-4)

    def test_twap_execution_reduces_var(self, mock_engine):
        """Verifica que la ejecución escalonada TWAP (5 días) reduce el VaR respecto a la conversión Bullet."""
        bullet = mock_engine.calculate_historical_var_es(confidence=0.99)
        twap = mock_engine.calculate_twap_benchmark(confidence=0.99)

        assert twap["twap_var_eur"] < bullet["var_eur"], "La ejecución escalonada TWAP debe reducir el VaR"
        assert twap["var_reduction_pct"] > 15.0, "La reducción de riesgo por diversificación temporal debe ser significativa"
        assert twap["twap_es_eur"] < bullet["es_eur"], "El Expected Shortfall bajo TWAP debe ser menor que bajo Bullet"

    def test_hedging_benchmark_forward_cost(self, mock_engine):
        """Verifica el cálculo del coste de cobertura vía FX Outright Forward."""
        hedge = mock_engine.calculate_hedging_benchmark(rate_diff_annual_bps=150.0, bid_ask_spread_bps=1.1)
        assert hedge["total_hedge_cost_eur"] > 0
        assert hedge["total_hedge_cost_eur"] < 100.0, "El coste de un Forward a 5d sobre 100k USD es de pocas decenas de euros"
        assert hedge["cost_to_es_ratio_pct"] < 5.0, "El coste del Forward debe ser inferior al 5% del riesgo de cola ES"

    def test_ewma_parametric_var(self, mock_engine):
        """Verifica consistencia del benchmark paramétrico EWMA."""
        ewma_res = mock_engine.calculate_ewma_var_es(confidence=0.99)
        assert ewma_res["var_eur"] > 0
        assert ewma_res["es_eur"] >= ewma_res["var_eur"]
        assert ewma_res["vol_annualized_pct"] > 0

    def test_traffic_light_structure(self, mock_engine):
        """Verifica que el semáforo devuelva estados admisibles y métricas lógicas."""
        tl = mock_engine.compute_traffic_light()
        assert tl["status"] in ["VERDE", "AMARILLO", "ROJO"]
        assert 0 <= tl["percentile"] <= 100
        assert tl["current_vol_annualized_pct"] > 0

    def test_kupiec_and_christoffersen_statistical_logic(self, mock_engine):
        """
        Verifica los contrastes de Kupiec (POF) y Christoffersen (Independencia y Cobertura Condicional):
        1. Secuencia calibrada y dispersa (10 fallos aislados en 1000 días): acepta POF, IND y CC.
        2. Secuencia con clustering extremo (10 fallos todos consecutivos): acepta POF pero RECHAZA IND y CC.
        """
        # Caso 1: 10 fallos aislados en 1000 ensayos (n11 = 0)
        b_dispersed = np.zeros(1000, dtype=int)
        b_dispersed[np.arange(50, 1000, 100)] = 1  # 10 fallos separados cada 100 días
        stats_disp = FXRiskEngine.compute_kupiec_and_christoffersen(b_dispersed, alpha=0.01)

        assert stats_disp["kupiec_accepted"] is True
        assert stats_disp["n11"] == 0
        assert stats_disp["christoffersen_p_ind"] > 0.05
        assert stats_disp["cc_accepted"] is True

        # Caso 2: 10 fallos todos seguidos en bloque (n11 = 9)
        b_clustered = np.zeros(1000, dtype=int)
        b_clustered[100:110] = 1
        stats_clust = FXRiskEngine.compute_kupiec_and_christoffersen(b_clustered, alpha=0.01)

        assert stats_clust["kupiec_accepted"] is True, "Kupiec no detecta el clustering porque solo mira el conteo total (10/1000)"
        assert stats_clust["n11"] == 9
        assert stats_clust["christoffersen_p_ind"] < 0.001, "Christoffersen debe detectar y rechazar el clustering de fallos"
        assert stats_clust["cc_accepted"] is False

        # Verificar ejecución del motor en modo solapado (step=1) y no solapado (step=5)
        bt_1 = mock_engine.run_backtest(window=500, step=1)
        bt_5 = mock_engine.run_backtest(window=500, step=5)
        assert bt_1["total_evaluations"] > bt_5["total_evaluations"]
