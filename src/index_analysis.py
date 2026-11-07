"""
Módulo de análisis cuantitativo de índices bursátiles globales y correlaciones en regímenes de estrés (VIX).
Caso Práctico FlatexDEGIRO - Tarea 2: Index Analysis.

Investiga empíricamente el fenómeno de "Correlation Breakdown", asimetría temporal,
outliers estructurales (VIX) y colas pesadas a lo largo de 18 años de datos de mercado (2008-2026).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética para gráficos de calidad editorial
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


@dataclass
class RegimeConfig:
    """Umbrales cuantitativos para definición de regímenes de mercado."""
    vix_calm_threshold: float = 20.0       # VIX < 20: Régimen calmo / complacencia
    vix_distress_threshold: float = 30.0   # VIX >= 30: Régimen de estrés / pánico
    vix_extreme_threshold: float = 40.0    # VIX >= 40: Crisis sistémica extrema
    sp500_tail_quantile: float = 0.05      # Percentil 5% de peores caídas del S&P 500


REGION_MAPPING = {
    "US": "North America",
    "DE": "Europe Core",
    "FR": "Europe Core",
    "NL": "Europe Core",
    "BE": "Europe Core",
    "EU": "Europe Core",
    "CH": "Europe Core",
    "GB": "Europe Core",
    "AT": "Europe Core",
    "FI": "Europe Core",
    "SE": "Europe Core",
    "ES": "Europe Periphery",
    "PL": "Europe East",
    "HU": "Europe East",
    "CZ": "Europe East",
    "JP": "Asia-Pacific",
    "HK": "Asia-Pacific",
    "TW": "Asia-Pacific",
    "KR": "Asia-Pacific",
    "AU": "Asia-Pacific",
    "IN": "Asia-Pacific (Emerging)",
}


class IndexDataLoader:
    """
    Carga, valida y alinea series temporales de índices bursátiles y metadatos.
    Resuelve específicamente la duplicidad de asset_key en US6311011026 (IXIC vs NDX).
    """

    def __init__(self, dim_path: Path, prices_path: Path):
        self.dim_path = Path(dim_path)
        self.prices_path = Path(prices_path)

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carga dim_master y price_hist, limpia duplicidades y computa retornos logarítmicos.

        Returns:
            Tuple con:
                - dim_df: Metadatos enriquecidos con región.
                - prices_pivot: Matriz de precios ajustados (Fecha x Ticker).
                - returns_pivot: Matriz de retornos logarítmicos diarios de renta variable.
        """
        dim = pd.read_csv(self.dim_path)
        prices = pd.read_parquet(self.prices_path)

        # Enriquecer dim con regiones macroeconómicas
        dim["region"] = dim["country_iso2"].map(REGION_MAPPING).fillna("Other")

        # ---------------------------------------------------------------------
        # Resolución de anomalía de datos en US6311011026 (NASDAQ Composite vs NASDAQ 100)
        # En price_hist.parquet, las filas de este asset_key están intercaladas de forma
        # contigua: filas pares corresponden a IXIC (Composite) e impares a NDX (Nasdaq 100).
        # ---------------------------------------------------------------------
        nasdaq_mask = prices["asset_key"] == "US6311011026"
        p_nasdaq = prices[nasdaq_mask].copy()
        p_other = prices[~nasdaq_mask].copy()

        # Mapear tickers estándar para los 29 activos restantes
        key_to_ticker = dim.set_index("asset_key")["ticker"].to_dict()
        p_other["ticker"] = p_other["asset_key"].map(key_to_ticker)

        # Separar filas intercaladas de NASDAQ
        nasdaq_ixic = p_nasdaq.iloc[0::2].copy()
        nasdaq_ixic["ticker"] = "IXIC.INDX"

        nasdaq_ndx = p_nasdaq.iloc[1::2].copy()
        nasdaq_ndx["ticker"] = "NDX.INDX"

        clean_prices = pd.concat([p_other, nasdaq_ixic, nasdaq_ndx], ignore_index=True)

        # Pivotar precios ajustados ordenados cronológicamente
        prices_pivot = clean_prices.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()

        # Separar activos de renta variable de la serie de volatilidad implícita (VIX)
        equity_cols = [c for c in prices_pivot.columns if c != "VIX.INDX"]
        returns_pivot = np.log(prices_pivot[equity_cols] / prices_pivot[equity_cols].shift(1))

        return dim, prices_pivot, returns_pivot


class IndexRiskAnalyzer:
    """
    Motor cuantitativo de correlaciones condicionales, outliers y fat tails en distress.
    """

    def __init__(
        self,
        dim_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        regime_config: Optional[RegimeConfig] = None,
    ):
        self.dim = dim_df
        self.prices = prices_df
        self.returns = returns_df
        self.config = regime_config or RegimeConfig()
        self.vix = prices_df["VIX.INDX"].dropna()

    def get_regime_dates(self) -> Dict[str, pd.Index]:
        """Identifica los conjuntos de fechas correspondientes a cada régimen de mercado."""
        calm_dates = self.vix[self.vix < self.config.vix_calm_threshold].index
        distress_dates = self.vix[self.vix >= self.config.vix_distress_threshold].index
        extreme_dates = self.vix[self.vix >= self.config.vix_extreme_threshold].index

        # Régimen de cola del S&P 500 (retornos <= percentil 5%)
        sp500_ret = self.returns["GSPC.INDX"].dropna()
        q05 = sp500_ret.quantile(self.config.sp500_tail_quantile)
        sp_tail_dates = sp500_ret[sp500_ret <= q05].index

        return {
            "calm": calm_dates,
            "distress": distress_dates,
            "extreme": extreme_dates,
            "sp500_tail": sp_tail_dates,
        }

    def compute_regime_correlations(self) -> Dict[str, Any]:
        """
        Calcula las matrices de correlación de Pearson condicionales al régimen.
        Devuelve C_calm, C_distress y Delta_C = C_distress - C_calm.
        """
        regimes = self.get_regime_dates()
        calm_idx = self.returns.index.intersection(regimes["calm"])
        dist_idx = self.returns.index.intersection(regimes["distress"])

        corr_calm = self.returns.loc[calm_idx].corr()
        corr_distress = self.returns.loc[dist_idx].corr()
        corr_diff = corr_distress - corr_calm

        # Correlación media entre pares (triángulo superior estricto)
        def mean_upper_triangle(mat: pd.DataFrame) -> float:
            vals = mat.values
            n = len(vals)
            upper = vals[np.triu_indices(n, k=1)]
            return float(np.nanmean(upper))

        mean_calm = mean_upper_triangle(corr_calm)
        mean_dist = mean_upper_triangle(corr_distress)
        mean_shift = mean_dist - mean_calm
        pct_increase = (mean_shift / mean_calm) * 100

        return {
            "corr_calm": corr_calm,
            "corr_distress": corr_distress,
            "corr_diff": corr_diff,
            "mean_corr_calm": mean_calm,
            "mean_corr_distress": mean_dist,
            "mean_corr_shift": mean_shift,
            "pct_increase": pct_increase,
            "calm_days_count": len(calm_idx),
            "distress_days_count": len(dist_idx),
        }

    def compute_regional_block_shifts(self) -> pd.DataFrame:
        """
        Calcula el cambio de correlación condicional desglosado por bloques geográficos y geopolíticos.
        """
        corr_res = self.compute_regime_correlations()
        c_calm = corr_res["corr_calm"]
        c_dist = corr_res["corr_distress"]

        us_tickers = ["GSPC.INDX", "DJI.INDX", "IXIC.INDX", "NDX.INDX"]
        eu_core = ["GDAXI.INDX", "STOXX50E.INDX", "FCHI.INDX", "AEX.INDX", "SXXP.INDX", "BUK100P.INDX", "SSMI.INDX"]
        eu_east = ["WIG20.INDX", "BUX.INDX", "PX.INDX"]
        asia = ["N225.INDX", "TOPX.INDX", "HSI.INDX", "KS11.INDX", "TWII.INDX", "AXJO.INDX"]
        india = ["BSESN.INDX", "NSEI.INDX"]

        blocks = [
            ("Intra-US", us_tickers, None),
            ("Intra-Europe Core", eu_core, None),
            ("US vs Europe Core", us_tickers, eu_core),
            ("Intra-Asia", asia, None),
            ("US vs Asia (Contemporáneo)", us_tickers, asia),
            ("Europe Core vs Asia", eu_core, asia),
            ("Europe Core vs Eastern Europe", eu_core, eu_east),
            ("Intra-Eastern Europe", eu_east, None),
            ("India vs US", india, us_tickers),
            ("India vs Europe Core", india, eu_core),
        ]

        def get_block_avg(mat: pd.DataFrame, t1: List[str], t2: Optional[List[str]]) -> float:
            if t2 is None:
                sub = mat.loc[t1, t1].values
                n = len(sub)
                return float(np.nanmean(sub[np.triu_indices(n, k=1)]))
            else:
                sub = mat.loc[t1, t2].values
                return float(np.nanmean(sub))

        records = []
        for name, b1, b2 in blocks:
            c_c = get_block_avg(c_calm, b1, b2)
            c_d = get_block_avg(c_dist, b1, b2)
            shift = c_d - c_c
            records.append({
                "block": name,
                "corr_calm": c_c,
                "corr_distress": c_d,
                "corr_shift": shift,
                "pct_increase": (shift / c_c) * 100 if c_c > 0 else np.nan,
            })

        return pd.DataFrame(records)

    def analyze_timezone_effect(self) -> Dict[str, Any]:
        """
        Demuestra empíricamente el sesgo del desfase horario (Timezone Asynchrony):
        Los mercados asiáticos cierran antes de la apertura de EE. UU.
        La correlación contemporánea es artificialmente baja, pero la correlación desfasada (t+1) se dispara.
        """
        sp500 = self.returns["GSPC.INDX"]
        asian_tickers = ["N225.INDX", "TOPX.INDX", "HSI.INDX", "KS11.INDX", "TWII.INDX", "AXJO.INDX"]

        results = []
        for t in asian_tickers:
            asia_series = self.returns[t]
            c_contemp = float(sp500.corr(asia_series))
            c_lag1 = float(asia_series.shift(-1).corr(sp500))  # Asia mañana vs EE. UU. hoy
            results.append({
                "ticker": t,
                "asset_name": self.dim.set_index("ticker").loc[t, "asset_name"],
                "corr_contemporaneous": c_contemp,
                "corr_lag1_response": c_lag1,
                "lag_gain": c_lag1 - c_contemp,
            })

        return pd.DataFrame(results)

    def compute_outliers_and_fat_tails(self) -> pd.DataFrame:
        """
        Calcula momentos estadísticos superiores (asimetría, curtosis excesiva) y peores caídas diarias.
        Identifica la naturaleza no normal (leptocúrtica) de los índices bursátiles.
        """
        stats = []
        for ticker in self.returns.columns:
            r = self.returns[ticker].dropna()
            meta = self.dim.set_index("ticker").loc[ticker]
            min_date = r.idxmin()
            max_date = r.idxmax()

            stats.append({
                "ticker": ticker,
                "country": meta["country_iso2"],
                "region": meta["region"],
                "asset_name": meta["asset_name"],
                "mean_annualized_pct": float(r.mean() * 252 * 100),
                "vol_annualized_pct": float(r.std() * np.sqrt(252) * 100),
                "skewness": float(skew(r)),
                "excess_kurtosis": float(kurtosis(r)),
                "worst_1d_drop_pct": float(r.min() * 100),
                "worst_1d_date": min_date,
                "best_1d_gain_pct": float(r.max() * 100),
                "best_1d_date": max_date,
            })

        return pd.DataFrame(stats).sort_values("excess_kurtosis", ascending=False).reset_index(drop=True)

    def analyze_vix_structural_outlier(self) -> Dict[str, Any]:
        """
        Analiza el VIX como activo singular: relación inversa con la renta variable,
        asimetría de apalancamiento (*leverage effect*) y no estacionariedad.
        """
        vix_ret = np.log(self.prices["VIX.INDX"] / self.prices["VIX.INDX"].shift(1)).dropna()
        sp_ret = self.returns["GSPC.INDX"].dropna()
        common_idx = sp_ret.index.intersection(vix_ret.index)

        corr_sp_vix = float(sp_ret.loc[common_idx].corr(vix_ret.loc[common_idx]))

        # Retorno medio del VIX en días positivos del S&P vs días negativos
        up_days = common_idx[sp_ret.loc[common_idx] > 0]
        down_days = common_idx[sp_ret.loc[common_idx] < 0]

        avg_vix_change_on_up = float(vix_ret.loc[up_days].mean() * 100)
        avg_vix_change_on_down = float(vix_ret.loc[down_days].mean() * 100)

        return {
            "corr_sp500_vix": corr_sp_vix,
            "avg_vix_on_sp500_up_pct": avg_vix_change_on_up,
            "avg_vix_on_sp500_down_pct": avg_vix_change_on_down,
            "asymmetry_ratio": abs(avg_vix_change_on_down / avg_vix_change_on_up) if avg_vix_change_on_up != 0 else np.nan,
            "vix_historical_min": float(self.vix.min()),
            "vix_historical_max": float(self.vix.max()),
            "vix_historical_median": float(self.vix.median()),
        }


class IndexVisualizer:
    """
    Genera visualizaciones de alta calidad analítica para su inclusión en informes y presentación.
    """

    def __init__(self, analyzer: IndexRiskAnalyzer, output_dir: Path = Path("figures")):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_correlation_diff_heatmap(self) -> Path:
        """Figura 1: Heatmap diferencial (Correlación Estrés - Correlación Calma)."""
        corr_res = self.analyzer.compute_regime_correlations()
        diff = corr_res["corr_diff"]

        # Ordenar por región para evidenciar bloques geográficos
        order = self.analyzer.dim.sort_values(by=["region", "country_iso2"])["ticker"].values
        ordered_tickers = [t for t in order if t in diff.index]
        diff_ordered = diff.loc[ordered_tickers, ordered_tickers]

        plt.figure(figsize=(14, 11))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            diff_ordered,
            cmap="vlag",
            center=0.0,
            vmin=-0.2,
            vmax=0.5,
            cbar_kws={"label": "Cambio en Coeficiente de Correlación (Δρ = ρ_estrés - ρ_calma)"},
            linewidths=0.3,
            linecolor="#F1F5F9",
        )
        plt.title(
            f"El Fenómeno de 'Correlation Breakdown' en Mercados Globales\n"
            f"Incremento de Correlación en Régimen de Estrés (VIX ≥ 30) vs Calma (VIX < 20) | Promedio: +{corr_res['mean_corr_shift']:.3f} (+{corr_res['pct_increase']:.1f}%)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        out_path = self.output_dir / "01_correlation_diff_heatmap.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_regional_shift_bars(self) -> Path:
        """Figura 2: Desglose del incremento de correlación por bloques regionales."""
        df_blocks = self.analyzer.compute_regional_block_shifts().sort_values("corr_shift", ascending=True)

        fig, ax = plt.subplots(figsize=(11, 7))
        y = np.arange(len(df_blocks))
        height = 0.35

        ax.barh(y - height / 2, df_blocks["corr_calm"], height, label="Calma (VIX < 20)", color="#38BDF8", alpha=0.9)
        ax.barh(y + height / 2, df_blocks["corr_distress"], height, label="Estrés (VIX ≥ 30)", color="#EF4444", alpha=0.9)

        ax.set_yticks(y)
        ax.set_yticklabels(df_blocks["block"], fontsize=10, fontweight="medium")
        ax.set_xlabel("Coeficiente de Correlación de Pearson Medio", fontsize=11, fontweight="bold")
        ax.set_title(
            "Pérdida de Diversificación Internacional por Bloques Regionales\nComparativa Calma vs Estrés de Mercado",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.legend(frameon=True, facecolor="white", loc="lower right")

        # Añadir etiquetas con el cambio absoluto
        for idx, row in df_blocks.reset_index().iterrows():
            ax.annotate(
                f"Δ {row['corr_shift']:+.2f}",
                xy=(row["corr_distress"] + 0.02, idx + height / 2),
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#DC2626",
            )

        ax.set_xlim(0, 1.1)
        plt.tight_layout()
        out_path = self.output_dir / "02_regional_correlation_shift.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_rolling_correlation_vs_vix(self) -> Path:
        """Figura 3: Evolución temporal de la correlación móvil S&P 500 vs DAX comparada con el VIX."""
        sp500 = self.analyzer.returns["GSPC.INDX"]
        dax = self.analyzer.returns["GDAXI.INDX"]
        vix = self.analyzer.prices["VIX.INDX"]

        rolling_corr = sp500.rolling(60).corr(dax)
        df_ts = pd.DataFrame({"rolling_corr": rolling_corr, "vix": vix}).dropna()
        df_ts.index = pd.to_datetime(df_ts.index)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})

        # Eje 1: Correlación Móvil
        ax1.plot(df_ts.index, df_ts["rolling_corr"], color="#2563EB", lw=1.5, label="Correlación Móvil 60d (S&P 500 vs DAX)")
        ax1.axhline(0.5, color="#64748B", linestyle=":", lw=1, alpha=0.7)
        ax1.axhline(df_ts["rolling_corr"].mean(), color="#059669", linestyle="--", lw=1.2, label=f"Media Histórica ({df_ts['rolling_corr'].mean():.2f})")
        ax1.set_ylabel("Correlación Móvil (60 días)", fontsize=11, fontweight="bold")
        ax1.set_title("Co-movimiento Transatlántico (S&P 500 vs DAX) y Régimen de Volatilidad VIX (2008-2026)", fontsize=13, fontweight="bold")
        ax1.legend(loc="lower right", frameon=True)

        # Sombrear periodos de crisis severas
        distress_periods = df_ts[df_ts["vix"] >= 30].index
        ax1.set_ylim(-0.2, 1.0)

        # Eje 2: Nivel del VIX
        ax2.plot(df_ts.index, df_ts["vix"], color="#DC2626", lw=1.2, label="Índice VIX")
        ax2.axhline(30, color="#B91C1C", linestyle="--", lw=1.2, label="Umbral Estrés (VIX=30)")
        ax2.axhline(20, color="#15803D", linestyle=":", lw=1, label="Umbral Calma (VIX=20)")
        ax2.fill_between(df_ts.index, 30, df_ts["vix"], where=(df_ts["vix"] >= 30), color="#FCA5A5", alpha=0.5)
        ax2.set_ylabel("VIX", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Año", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        out_path = self.output_dir / "03_rolling_correlation_sp500_dax_vs_vix.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_fat_tails_and_drawdowns(self) -> Path:
        """Figura 4: Curtosis excesiva y peores caídas diarias en 1 día."""
        df_stats = self.analyzer.compute_outliers_and_fat_tails().head(12)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Curtosis Excesiva (Fat Tails)
        ax1.barh(df_stats["ticker"], df_stats["excess_kurtosis"], color="#8B5CF6", alpha=0.85)
        ax1.axvline(0, color="black", lw=0.8)
        ax1.set_xlabel("Curtosis Excesiva (Normal = 0)", fontsize=10, fontweight="bold")
        ax1.set_title("Colas Pesadas Extremas (Leptocurtosis)", fontsize=11, fontweight="bold")
        ax1.invert_yaxis()

        # Subplot 2: Peor Caída Diaria en 1 día (%)
        ax2.barh(df_stats["ticker"], df_stats["worst_1d_drop_pct"], color="#EF4444", alpha=0.85)
        ax2.set_xlabel("Peor Caída en 1 Sesión (%)", fontsize=10, fontweight="bold")
        ax2.set_title("Máximos Shocks Diarios Observados (2008-2026)", fontsize=11, fontweight="bold")
        ax2.invert_yaxis()

        for idx, row in df_stats.iterrows():
            date_str = str(row["worst_1d_date"])[:10]
            ax2.annotate(
                f"{date_str}",
                xy=(row["worst_1d_drop_pct"] - 0.5, idx),
                va="center",
                fontsize=8,
                color="#475569",
            )

        plt.suptitle("Outliers Estadísticos y Vulnerabilidad de Cola por Índice", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = self.output_dir / "04_fat_tails_kurtosis_drawdowns.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_timezone_lag_effect(self) -> Path:
        """Figura 5: Efecto del desfase horario en índices asiáticos."""
        df_tz = self.analyzer.analyze_timezone_effect()

        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(df_tz))
        width = 0.35

        ax.bar(x - width / 2, df_tz["corr_contemporaneous"], width, label="Correlación Contemporánea (t)", color="#94A3B8")
        ax.bar(x + width / 2, df_tz["corr_lag1_response"], width, label="Correlación Desfasada (Asia t+1 vs S&P 500 t)", color="#0284C7")

        ax.set_xticks(x)
        ax.set_xticklabels(df_tz["ticker"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Correlación con S&P 500", fontsize=11, fontweight="bold")
        ax.set_title("El Efecto del Desfase Horario: Desacople Aparente vs Transmisión Real", fontsize=12, fontweight="bold", pad=12)
        ax.legend(frameon=True, facecolor="white")

        for idx, row in df_tz.iterrows():
            ax.annotate(
                f"+{row['lag_gain']:.2f}",
                xy=(idx + width / 2, row["corr_lag1_response"] + 0.02),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#0369A1",
            )

        ax.set_ylim(0, 0.7)
        plt.tight_layout()
        out_path = self.output_dir / "05_timezone_lag_effect.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def generate_all_visualizations(self) -> List[Path]:
        """Ejecuta y guarda todas las figuras analíticas."""
        paths = [
            self.plot_correlation_diff_heatmap(),
            self.plot_regional_shift_bars(),
            self.plot_rolling_correlation_vs_vix(),
            self.plot_fat_tails_and_drawdowns(),
            self.plot_timezone_lag_effect(),
        ]
        return paths
