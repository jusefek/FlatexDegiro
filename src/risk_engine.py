"""
Motor cuantitativo de cálculo de riesgo de mercado FX (FX Risk Engine).
Implementa Simulación Histórica con revalorización completa (Through-the-Cycle y Point-in-Time),
Nivel Crítico de Spot (Trigger Rate), Benchmark de Ejecución Operativa (TWAP 5d),
Benchmark de Cobertura (FX Outright Forward), Benchmark Paramétrico EWMA y validación
regulatoria mediante los Tests de Kupiec (POF) y Christoffersen (Independencia y Cobertura Condicional)
en ventanas tanto solapadas como en bloques disjuntos no solapados.
"""
from dataclasses import dataclass
from datetime import datetime, date
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.config import config
from src.database import DuckDBManager

logger = logging.getLogger("FXRiskEngine")


@dataclass
class PositionConfig:
    """Parámetros de la posición en tesorería."""
    usd_amount: float = 100_000.0
    horizon_days: int = 5
    confidence_level: float = 0.99
    ewma_lambda: float = 0.94


class FXRiskEngine:
    """
    Motor cuantitativo de cálculo y backtesting de riesgo de tipo de cambio para tesorería.
    
    Convención de cotización del BCE:
    S = USD/EUR (Dólares por 1 Euro).
    Valor patrimonial en Euros: V_EUR = USD_AMOUNT / S.
    Riesgo: Depreciación del USD frente al EUR (subidas de S), lo que reduce los EUR recibidos.
    """

    def __init__(self, db_manager: Optional[DuckDBManager] = None, pos_config: Optional[PositionConfig] = None):
        self.db = db_manager or DuckDBManager()
        self.config = pos_config or PositionConfig()
        self._df_rates: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Extrae la serie histórica completa de tipos de cambio desde DuckDB en modo lectura."""
        query = """
        SELECT rate_date, spot_rate, ingested_at 
        FROM ecb_fx_rates 
        ORDER BY rate_date ASC;
        """
        with self.db.get_connection() as conn:
            df = conn.execute(query).df()

        if df.empty:
            raise ValueError("La tabla 'ecb_fx_rates' en DuckDB está vacía. Ejecute primero 'etl_ecb.py --mode initial'.")

        df["rate_date"] = pd.to_datetime(df["rate_date"]).dt.date
        self._df_rates = df
        logger.info(f"Cargadas {len(df)} cotizaciones históricas del BCE [{df['rate_date'].min()} a {df['rate_date'].max()}].")
        return self._df_rates

    @property
    def rates(self) -> pd.DataFrame:
        if self._df_rates is None:
            self.load_data()
        return self._df_rates

    def get_spot_context(self) -> Dict[str, Any]:
        """Obtiene la cotización spot actual y el valor patrimonial de la posición en EUR."""
        df = self.rates
        latest_row = df.iloc[-1]
        spot_rate = float(latest_row["spot_rate"])
        spot_date = latest_row["rate_date"]
        ingested_at = latest_row["ingested_at"]

        # Valor actual en EUR: 100,000 / S
        current_v_eur = self.config.usd_amount / spot_rate

        return {
            "spot_date": spot_date,
            "spot_rate": spot_rate,
            "usd_amount": self.config.usd_amount,
            "current_v_eur": current_v_eur,
            "ingested_at": ingested_at,
        }

    def compute_empirical_returns(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula retornos logarítmicos empíricos a 1 día y a h días hábiles sobre el histórico.
        
        Returns:
            Tuple con (retornos_1d, retornos_hd)
        """
        spots = self.rates["spot_rate"].values
        h = self.config.horizon_days
        # Retornos logarítmicos continuos a 1 día: ln(S_t / S_{t-1})
        rets_1d = np.log(spots[1:] / spots[:-1])
        # Retornos logarítmicos continuos a h días: ln(S_t / S_{t-h})
        rets_hd = np.log(spots[h:] / spots[:-h])
        return rets_1d, rets_hd

    def calculate_historical_var_es(self, confidence: Optional[float] = None, pit_window: int = 500) -> Dict[str, Any]:
        """
        Calcula el VaR y Expected Shortfall mediante Simulación Histórica con revalorización no lineal completa.
        Incluye tanto la métrica Through-the-Cycle (TTC, muestra completa) como Point-in-Time (PiT, últimos 500 días)
        y los niveles críticos de tipo de cambio (Trigger Rates) donde se consume el presupuesto de riesgo.
        """
        conf = confidence or self.config.confidence_level
        ctx = self.get_spot_context()
        S0 = ctx["spot_rate"]
        V0 = ctx["current_v_eur"]
        _, rets_5d = self.compute_empirical_returns()

        # Revalorización completa bajo los escenarios históricos (Through-the-Cycle)
        simulated_spots = S0 * np.exp(rets_5d)
        simulated_values_eur = self.config.usd_amount / simulated_spots
        simulated_losses_eur = V0 - simulated_values_eur  # Pérdida expresada como número positivo
        simulated_pnl_eur = simulated_values_eur - V0     # P&L tradicional (negativo = pérdida)

        # VaR empírico al percentil solicitado (Through-the-Cycle)
        var_eur = float(np.percentile(simulated_losses_eur, conf * 100))
        var_pct = float(var_eur / V0 * 100)

        # Expected Shortfall (CVaR): Media condicional de las pérdidas en la cola >= VaR
        tail_losses = simulated_losses_eur[simulated_losses_eur >= var_eur]
        es_eur = float(np.mean(tail_losses))
        es_pct = float(es_eur / V0 * 100)

        # Peor pérdida histórica observada (Stress test histórico extremo)
        worst_loss_eur = float(np.max(simulated_losses_eur))
        worst_loss_pct = float(worst_loss_eur / V0 * 100)

        # Nivel Crítico de Cotización (Trigger Rate): S_trigger = USD_AMOUNT / (V_0 - VaR)
        trigger_spot_var = float(self.config.usd_amount / (V0 - var_eur))
        trigger_spot_es = float(self.config.usd_amount / (V0 - es_eur))
        trigger_spot_move_pct = float((trigger_spot_var / S0 - 1.0) * 100)

        # Estimación Point-in-Time (PiT) sobre ventana móvil reciente (ej. 500 sesiones)
        eff_window = min(pit_window, len(simulated_losses_eur))
        pit_losses = simulated_losses_eur[-eff_window:]
        pit_var_eur = float(np.percentile(pit_losses, conf * 100))
        pit_var_pct = float(pit_var_eur / V0 * 100)
        pit_es_eur = float(np.mean(pit_losses[pit_losses >= pit_var_eur]))
        pit_es_pct = float(pit_es_eur / V0 * 100)

        return {
            "method": "Simulación Histórica (Full Revaluation - TTC)",
            "horizon_days": self.config.horizon_days,
            "confidence_level": conf,
            "var_eur": var_eur,
            "var_pct": var_pct,
            "es_eur": es_eur,
            "es_pct": es_pct,
            "worst_loss_eur": worst_loss_eur,
            "worst_loss_pct": worst_loss_pct,
            "trigger_spot_var": trigger_spot_var,
            "trigger_spot_es": trigger_spot_es,
            "trigger_spot_move_pct": trigger_spot_move_pct,
            "pit_window": eff_window,
            "pit_var_eur": pit_var_eur,
            "pit_var_pct": pit_var_pct,
            "pit_es_eur": pit_es_eur,
            "pit_es_pct": pit_es_pct,
            "total_scenarios": len(rets_5d),
            "simulated_losses_eur": simulated_losses_eur,
            "simulated_pnl_eur": simulated_pnl_eur,
        }

    def calculate_twap_benchmark(self, confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        Modela una estrategia de ejecución operativa escalonada (TWAP - Time-Weighted Average Price),
        convirtiendo USD_AMOUNT / h en cada uno de los próximos h días hábiles (t+1, ..., t+h),
        y cuantifica la reducción de VaR y Expected Shortfall frente a la conversión 'Bullet' en el día h.
        """
        conf = confidence or self.config.confidence_level
        ctx = self.get_spot_context()
        S0 = ctx["spot_rate"]
        V0 = ctx["current_v_eur"]
        spots = self.rates["spot_rate"].values
        h = self.config.horizon_days

        n_scen = len(spots) - h
        tranche_usd = self.config.usd_amount / float(h)
        twap_values_eur = np.zeros(n_scen)

        for j in range(1, h + 1):
            rel_change_j = spots[j : n_scen + j] / spots[:n_scen]
            sim_spot_j = S0 * rel_change_j
            twap_values_eur += tranche_usd / sim_spot_j

        twap_losses_eur = V0 - twap_values_eur
        twap_var_eur = float(np.percentile(twap_losses_eur, conf * 100))
        twap_var_pct = float(twap_var_eur / V0 * 100)
        twap_es_eur = float(np.mean(twap_losses_eur[twap_losses_eur >= twap_var_eur]))
        twap_es_pct = float(twap_es_eur / V0 * 100)

        bullet_metrics = self.calculate_historical_var_es(confidence=conf)
        var_saving_eur = bullet_metrics["var_eur"] - twap_var_eur
        var_reduction_pct = (var_saving_eur / bullet_metrics["var_eur"]) * 100 if bullet_metrics["var_eur"] > 0 else 0.0
        es_saving_eur = bullet_metrics["es_eur"] - twap_es_eur
        es_reduction_pct = (es_saving_eur / bullet_metrics["es_eur"]) * 100 if bullet_metrics["es_eur"] > 0 else 0.0

        return {
            "strategy": f"Conversión Escalonada TWAP ({h} tramos de USD {tranche_usd:,.0f}/día)",
            "tranche_usd": tranche_usd,
            "twap_var_eur": twap_var_eur,
            "twap_var_pct": twap_var_pct,
            "twap_es_eur": twap_es_eur,
            "twap_es_pct": twap_es_pct,
            "var_saving_eur": var_saving_eur,
            "var_reduction_pct": var_reduction_pct,
            "es_saving_eur": es_saving_eur,
            "es_reduction_pct": es_reduction_pct,
            "twap_losses_eur": twap_losses_eur,
        }

    def calculate_hedging_benchmark(
        self,
        rate_diff_annual_bps: float = 150.0,
        bid_ask_spread_bps: float = 1.1,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estima el coste financiero cierto de cubrir el 100% de la posición hoy mediante un
        contrato FX Outright Forward al horizonte h días:
        - Forward Points (Carry por diferencial de tipos SOFR vs €STR durante h/360 días).
        - Coste de ejecución por Bid-Ask Spread interbancario.
        """
        conf = confidence or self.config.confidence_level
        ctx = self.get_spot_context()
        S0 = ctx["spot_rate"]
        V0 = ctx["current_v_eur"]
        h = self.config.horizon_days

        # Coste de carry / puntos forward a h días (Convención ACT/360)
        carry_bps_period = rate_diff_annual_bps * (float(h) / 360.0)
        carry_cost_eur = float(V0 * (carry_bps_period / 10_000.0))

        # Coste de horquilla bid-ask institucional en EUR/USD
        spread_cost_eur = float(V0 * (bid_ask_spread_bps / 10_000.0))

        total_hedge_cost_eur = carry_cost_eur + spread_cost_eur
        total_hedge_cost_pct = float(total_hedge_cost_eur / V0 * 100)
        forward_rate = float(S0 * (1.0 + (carry_bps_period + bid_ask_spread_bps) / 10_000.0))

        hist_metrics = self.calculate_historical_var_es(confidence=conf)
        es_eur = hist_metrics["es_eur"]
        cost_to_es_ratio_pct = float(total_hedge_cost_eur / es_eur * 100) if es_eur > 0 else 0.0

        return {
            "instrument": f"FX Outright Forward ({h}d)",
            "forward_rate": forward_rate,
            "carry_cost_eur": carry_cost_eur,
            "spread_cost_eur": spread_cost_eur,
            "total_hedge_cost_eur": total_hedge_cost_eur,
            "total_hedge_cost_pct": total_hedge_cost_pct,
            "unhedged_es_eur": es_eur,
            "cost_to_es_ratio_pct": cost_to_es_ratio_pct,
        }

    def calculate_ewma_var_es(self, confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        Calcula el VaR y Expected Shortfall paramétrico utilizando volatilidad EWMA (RiskMetrics, lambda=0.94).
        Permite comparar el riesgo actual sensible a la volatilidad reciente frente a la distribución histórica.
        """
        conf = confidence or self.config.confidence_level
        lam = self.config.ewma_lambda
        ctx = self.get_spot_context()
        S0 = ctx["spot_rate"]
        V0 = ctx["current_v_eur"]
        rets_1d, _ = self.compute_empirical_returns()

        # Cálculo recursivo de la varianza EWMA: sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2
        variance_ewma = np.zeros(len(rets_1d))
        variance_ewma[0] = rets_1d[0] ** 2
        for t in range(1, len(rets_1d)):
            variance_ewma[t] = lam * variance_ewma[t - 1] + (1 - lam) * (rets_1d[t] ** 2)

        vol_1d_recent = float(np.sqrt(variance_ewma[-1]))
        # Escalado al horizonte de 5 días bajo convención de raíz del tiempo para modelo benchmark
        vol_5d_recent = float(vol_1d_recent * np.sqrt(self.config.horizon_days))
        vol_annualized = float(vol_1d_recent * np.sqrt(252))

        # Cuantil normal estándar
        z_score = float(norm.ppf(conf))

        # Revalorización paramétrica exacta con curvatura
        S_stressed = S0 * np.exp(z_score * vol_5d_recent)
        var_eur = float(V0 - (self.config.usd_amount / S_stressed))
        var_pct = float(var_eur / V0 * 100)

        # Expected Shortfall paramétrico normal: E[L | L >= VaR]
        es_factor = norm.pdf(z_score) / (1.0 - conf)
        S_stressed_es = S0 * np.exp(es_factor * vol_5d_recent)
        es_eur = float(V0 - (self.config.usd_amount / S_stressed_es))
        es_pct = float(es_eur / V0 * 100)

        return {
            "method": f"Paramétrico EWMA (RiskMetrics λ={lam})",
            "vol_1d_pct": vol_1d_recent * 100,
            "vol_5d_pct": vol_5d_recent * 100,
            "vol_annualized_pct": vol_annualized * 100,
            "var_eur": var_eur,
            "var_pct": var_pct,
            "es_eur": es_eur,
            "es_pct": es_pct,
        }

    def compute_traffic_light(self) -> Dict[str, Any]:
        """
        Determina el estado del semáforo de riesgo para el Management Board
        comparando la volatilidad realizada reciente (últimos 21 días) con los percentiles históricos.
        """
        rets_1d, _ = self.compute_empirical_returns()
        series_rets = pd.Series(rets_1d)
        rolling_vol = series_rets.rolling(21).std() * np.sqrt(252)
        valid_vols = rolling_vol.dropna()

        current_vol = float(valid_vols.iloc[-1])
        p60 = float(valid_vols.quantile(0.60))
        p85 = float(valid_vols.quantile(0.85))
        percentile = float((valid_vols < current_vol).mean() * 100)

        if current_vol <= p60:
            status = "VERDE"
            color = "#10B981"  # Emerald Green
            label = "Régimen Normal / Baja Volatilidad"
            recommendation = "Exposición cambiaria dentro de parámetros normales. Se sugiere ejecución escalonada TWAP a coste cero."
        elif current_vol <= p85:
            status = "AMARILLO"
            color = "#F59E0B"  # Amber Yellow
            label = "Atención Preventiva / Volatilidad Moderada"
            recommendation = "Incremento moderado de volatilidad. Monitorear nivel crítico S_VaR y considerar cobertura parcial."
        else:
            status = "ROJO"
            color = "#EF4444"  # Red
            label = "Alta Volatilidad / Mercado en Estrés"
            recommendation = "Régimen de volatilidad elevada. Cerrar exposición mediante FX Outright Forward inmediato."

        return {
            "status": status,
            "color": color,
            "label": label,
            "current_vol_annualized_pct": current_vol * 100,
            "historical_p60_pct": p60 * 100,
            "historical_p85_pct": p85 * 100,
            "percentile": percentile,
            "recommendation": recommendation,
        }

    @staticmethod
    def compute_kupiec_and_christoffersen(breaches: np.ndarray, alpha: float = 0.01) -> Dict[str, Any]:
        """
        Calcula formalmente:
        1. Test de Cobertura Incondicional de Kupiec (1995) - LR_POF ~ chi2(1)
        2. Test de Independencia de Markov de Christoffersen (1998) - LR_IND ~ chi2(1)
        3. Test de Cobertura Condicional Conjunta de Christoffersen - LR_CC = LR_POF + LR_IND ~ chi2(2)
        """
        b = np.asarray(breaches, dtype=int)
        N = len(b)
        x = int(np.sum(b))
        p = alpha
        obs_rate = x / N if N > 0 else 0.0

        # 1. Kupiec POF
        if x == 0:
            lr_pof = float(-2.0 * N * np.log(1.0 - p))
        elif x == N:
            lr_pof = float(-2.0 * N * np.log(p))
        else:
            term1 = x * np.log(x / (N * p))
            term2 = (N - x) * np.log((N - x) / (N * (1.0 - p)))
            lr_pof = float(2.0 * (term1 + term2))

        pval_pof = float(1.0 - chi2.cdf(lr_pof, df=1))

        # 2. Christoffersen Independence (Cadena de Markov de primer orden)
        b_prev = b[:-1]
        b_curr = b[1:]
        n00 = int(np.sum((b_prev == 0) & (b_curr == 0)))
        n01 = int(np.sum((b_prev == 0) & (b_curr == 1)))
        n10 = int(np.sum((b_prev == 1) & (b_curr == 0)))
        n11 = int(np.sum((b_prev == 1) & (b_curr == 1)))

        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi_all = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

        def _safe_loglik(count0: int, count1: int, prob1: float) -> float:
            ll = 0.0
            if count0 > 0 and prob1 < 1.0:
                ll += count0 * np.log(1.0 - prob1)
            if count1 > 0 and prob1 > 0.0:
                ll += count1 * np.log(prob1)
            return ll

        ll_null = _safe_loglik(n00 + n10, n01 + n11, pi_all)
        ll_alt = _safe_loglik(n00, n01, pi01) + _safe_loglik(n10, n11, pi11)
        lr_ind = float(max(0.0, -2.0 * (ll_null - ll_alt)))
        pval_ind = float(1.0 - chi2.cdf(lr_ind, df=1))

        # 3. Cobertura Condicional Conjunta (LR_CC = LR_POF + LR_IND)
        lr_cc = float(lr_pof + lr_ind)
        pval_cc = float(1.0 - chi2.cdf(lr_cc, df=2))

        return {
            "total_evaluations": N,
            "breaches_count": x,
            "observed_failure_rate": obs_rate * 100,
            "expected_failure_rate": p * 100,
            "kupiec_lr_stat": lr_pof,
            "kupiec_p_value": pval_pof,
            "kupiec_accepted": bool(pval_pof >= 0.05),
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "pi01_pct": pi01 * 100,
            "pi11_pct": pi11 * 100,
            "christoffersen_lr_ind": lr_ind,
            "christoffersen_p_ind": pval_ind,
            "christoffersen_lr_cc": lr_cc,
            "christoffersen_p_cc": pval_cc,
            "cc_accepted": bool(pval_cc >= 0.05),
        }

    def run_backtest(self, window: int = 500, confidence: Optional[float] = None, step: int = 1) -> Dict[str, Any]:
        """
        Ejecuta el motor de backtesting histórico móvil fuera de muestra (out-of-sample).
        Soporta:
        - step = 1: Ventanas diarias solapadas (Overlapping, muestra máxima pero induce estructura MA(h-1)).
        - step = h: Bloques disjuntos no solapados (Non-overlapping, elimina autocorrelación artificial).
        """
        conf = confidence or self.config.confidence_level
        alpha = 1.0 - conf
        spots = self.rates["spot_rate"].values
        dates = self.rates["rate_date"].values
        n_days = len(spots)
        usd_amount = self.config.usd_amount
        h = self.config.horizon_days

        eval_dates = []
        actual_losses = []
        var_forecasts = []
        breaches = []

        for t in range(window, n_days - h, step):
            hist_spots = spots[t - window : t + 1]
            rets_5d_t = np.log(hist_spots[h:] / hist_spots[:-h])

            v_t = usd_amount / spots[t]
            sim_spots_t = spots[t] * np.exp(rets_5d_t)
            sim_losses_t = v_t - (usd_amount / sim_spots_t)
            var_t = float(np.percentile(sim_losses_t, conf * 100))

            v_realized = usd_amount / spots[t + h]
            actual_loss_t = float(v_t - v_realized)

            is_breach = int(actual_loss_t > var_t)

            eval_dates.append(dates[t])
            actual_losses.append(actual_loss_t)
            var_forecasts.append(var_t)
            breaches.append(is_breach)

        stats = self.compute_kupiec_and_christoffersen(np.array(breaches), alpha=alpha)
        model_passed = stats["kupiec_accepted"]

        df_backtest = pd.DataFrame({
            "date": pd.to_datetime(eval_dates),
            "actual_loss_eur": actual_losses,
            "var_99_eur": var_forecasts,
            "is_breach": breaches,
        })

        stats.update({
            "step_days": step,
            "scheme_name": "Ventanas Móviles Solapadas (Overlapping, step=1d)" if step == 1 else f"Bloques Disjuntos No Solapados (Non-Overlapping, step={step}d)",
            "model_accepted": model_passed,
            "test_conclusion": "ACEPTADO (Kupiec p ≥ 0.05)" if model_passed else "RECHAZADO (Kupiec p < 0.05)",
            "df_backtest": df_backtest,
            "start_date": eval_dates[0] if eval_dates else None,
            "end_date": eval_dates[-1] if eval_dates else None,
        })
        return stats
