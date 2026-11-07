"""
Pruebas unitarias para el motor de análisis de índices globales y correlaciones (Tarea 2).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.index_analysis import IndexDataLoader, IndexRiskAnalyzer, RegimeConfig


class TestIndexAnalysis:
    """Verificación de carga de datos, cálculo de retornos y coherencia matemática de correlaciones."""

    @pytest.fixture(scope="class")
    def loaded_data(self):
        """Carga y procesa los datos reales una sola vez para la suite de pruebas."""
        dim_path = Path("dim_master.csv")
        prices_path = Path("price_hist.parquet")
        loader = IndexDataLoader(dim_path=dim_path, prices_path=prices_path)
        dim_df, prices_df, returns_df = loader.load_and_preprocess()
        return dim_df, prices_df, returns_df

    def test_duplicate_asset_key_resolved_cleanly(self, loaded_data):
        """
        Verifica que la duplicidad de asset_key US6311011026 fue resuelta:
        - IXIC.INDX y NDX.INDX existen como columnas independientes.
        - Existen exactamente 31 activos en total (30 de renta variable + 1 VIX).
        """
        dim_df, prices_df, returns_df = loaded_data

        assert "IXIC.INDX" in prices_df.columns, "IXIC.INDX debe estar presente"
        assert "NDX.INDX" in prices_df.columns, "NDX.INDX debe estar presente"
        assert "VIX.INDX" in prices_df.columns, "VIX.INDX debe estar presente"
        assert prices_df.shape[1] == 31, "Debe haber exactamente 31 tickers únicos"
        assert returns_df.shape[1] == 30, "Debe haber 30 índices de acciones en la matriz de retornos"

    def test_returns_matrix_validity(self, loaded_data):
        """Verifica que los retornos diarios no contengan valores infinitos ni superen rangos realistas."""
        _, _, returns_df = loaded_data

        clean_rets = returns_df.iloc[1:]
        # Verificar que no haya valores infinitos
        assert not np.isinf(clean_rets.values).any(), "No debe haber retornos infinitos"
        
        # Comprobar que los mínimos y máximos observados se sitúan en rangos económicos coherentes [-30%, +30%]
        min_ret = float(clean_rets.min().min())
        max_ret = float(clean_rets.max().max())
        assert min_ret > -0.30, f"Ningún índice debe caer más de un 30% en un día (observado: {min_ret:.2%})"
        assert max_ret < 0.30, f"Ningún índice debe subir más de un 30% en un día (observado: {max_ret:.2%})"

    def test_regime_classification_mutually_exclusive(self, loaded_data):
        """Verifica que los regímenes de calma (VIX < 20) y estrés (VIX >= 30) sean disjuntos."""
        dim_df, prices_df, returns_df = loaded_data
        analyzer = IndexRiskAnalyzer(dim_df=dim_df, prices_df=prices_df, returns_df=returns_df)
        regimes = analyzer.get_regime_dates()

        intersection = regimes["calm"].intersection(regimes["distress"])
        assert len(intersection) == 0, "Los regímenes de calma y estrés deben ser mutuamente excluyentes"
        assert len(regimes["calm"]) > 2000, "Debe haber un número representativo de días de calma"
        assert len(regimes["distress"]) > 300, "Debe haber un número representativo de días de estrés"

    def test_correlation_breakdown_hypothesis(self, loaded_data):
        """
        Contrasta la hipótesis empírica central del caso práctico:
        La correlación media en estrés DEBE ser significativamente superior a la correlación en calma.
        """
        dim_df, prices_df, returns_df = loaded_data
        analyzer = IndexRiskAnalyzer(dim_df=dim_df, prices_df=prices_df, returns_df=returns_df)
        corr_res = analyzer.compute_regime_correlations()

        c_calm = corr_res["corr_calm"]
        c_dist = corr_res["corr_distress"]

        # 1. Propiedades algebraicas de matrices de correlación
        assert np.allclose(c_calm.values, c_calm.values.T), "La matriz de calma debe ser simétrica"
        assert np.allclose(c_dist.values, c_dist.values.T), "La matriz de estrés debe ser simétrica"
        assert np.allclose(np.diag(c_calm.values), 1.0), "La diagonal en calma debe ser 1.0"
        assert np.allclose(np.diag(c_dist.values), 1.0), "La diagonal en estrés debe ser 1.0"

        # 2. Fenómeno de Correlation Breakdown
        assert corr_res["mean_corr_distress"] > corr_res["mean_corr_calm"], (
            f"La correlación media en estrés ({corr_res['mean_corr_distress']:.3f}) "
            f"debe ser mayor que en calma ({corr_res['mean_corr_calm']:.3f})"
        )
        assert corr_res["mean_corr_shift"] > 0.10, "El incremento de correlación debe ser de al menos +0.10"

    def test_vix_structural_inverse_relationship(self, loaded_data):
        """Verifica la correlación fuertemente negativa del VIX con la renta variable (S&P 500)."""
        dim_df, prices_df, returns_df = loaded_data
        analyzer = IndexRiskAnalyzer(dim_df=dim_df, prices_df=prices_df, returns_df=returns_df)
        vix_res = analyzer.analyze_vix_structural_outlier()

        assert vix_res["corr_sp500_vix"] < -0.60, "El VIX debe tener correlación negativa fuerte con S&P 500 (< -0.60)"
        assert vix_res["asymmetry_ratio"] > 1.0, "La reacción del VIX ante caídas debe ser mayor que ante subidas"
