"""Fatores de risco: correlação e Análise de Componentes Principais.

Para que serve no pipeline
--------------------------
A matriz de correlação mostra o quanto os ativos "andam juntos" — insumo
de diversificação. O PCA vai além: decompõe a covariância dos retornos em
componentes ortogonais ordenados por variância explicada. Em carteiras de
ações, o 1º componente costuma capturar o "fator mercado" (todas as
cargas com mesmo sinal); os seguintes, efeitos setoriais/estilo. Se
poucos componentes explicam quase tudo, a diversificação aparente da
carteira é menor do que o número de papéis sugere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .exceptions import DataValidationError
from .logging_setup import get_logger
from .validation import require_min_length

log = get_logger(__name__)


def correlation_matrix(returns: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Matriz de correlação de Pearson dos retornos.

    Parameters
    ----------
    returns:
        Retornos (datas x tickers).
    window:
        ``None`` usa a amostra completa; um inteiro usa apenas os
        últimos ``window`` pregões (retrato "móvel" da correlação —
        correlações mudam com o regime de mercado, e o retrato recente
        pode divergir muito da média histórica).

    Returns
    -------
    pandas.DataFrame
        Matriz simétrica (tickers x tickers) com diagonal 1.

    Raises
    ------
    DataValidationError
        Menos de 2 colunas, janela inválida ou histórico insuficiente.
    """
    if returns.shape[1] < 2:
        raise DataValidationError("Correlação exige pelo menos 2 ativos.")
    data = returns
    if window is not None:
        if window < 2:
            raise DataValidationError(f"window={window} inválido; use >= 2 ou None.")
        if len(returns) < window:
            raise DataValidationError(
                f"Histórico ({len(returns)}) menor que a janela de correlação ({window})."
            )
        data = returns.tail(window)
    corr = data.corr(method="pearson")
    log.info(
        "Correlação calculada (%s, %d observações).",
        "amostra completa" if window is None else f"janela={window}",
        len(data),
    )
    return corr


@dataclass(frozen=True)
class PcaResult:
    """Resultado da extração de fatores por PCA.

    Attributes
    ----------
    explained_variance_ratio:
        Fração da variância total explicada por componente (soma <= 1).
    loadings:
        Cargas: linhas = tickers, colunas = ``PC1..PCk``. A carga do
        ativo *i* no componente *j* diz o quanto *i* "participa" do
        fator *j* (autovetor da matriz de covariância).
    factor_returns:
        Séries temporais dos fatores (projeção dos retornos nos
        componentes): linhas = datas, colunas = ``PC1..PCk``.
    """

    explained_variance_ratio: pd.Series
    loadings: pd.DataFrame
    factor_returns: pd.DataFrame


def pca_factors(
    returns: pd.DataFrame,
    n_components: int = 3,
    standardize: bool = True,
) -> PcaResult:
    """Extrai fatores de risco por PCA sobre os retornos.

    Parameters
    ----------
    returns:
        Retornos (datas x tickers); linhas com NaN são descartadas
        (PCA exige matriz completa).
    n_components:
        Nº de componentes; limitado por ``min(n_ativos, n_datas)``.
    standardize:
        Se ``True`` (padrão), centraliza **e divide pelo desvio-padrão**
        de cada ativo antes do PCA — equivale a fazer PCA na matriz de
        *correlação*. Sem isso, ativos mais voláteis dominariam os
        componentes por pura escala, não por estrutura de co-movimento.

    Returns
    -------
    PcaResult
        Variância explicada, cargas e retornos dos fatores.

    Raises
    ------
    DataValidationError
        Dados insuficientes ou ``n_components`` maior que o posto possível.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> base = rng.normal(0, 0.01, 300)
    >>> df = pd.DataFrame({"A": base, "B": base * 1.0000001})
    >>> res = pca_factors(df, n_components=1)
    >>> res.explained_variance_ratio.iloc[0] > 0.99
    True
    """
    clean = returns.dropna(how="any")
    require_min_length(clean, max(30, returns.shape[1] + 1), "retornos (PCA)")
    max_components = min(clean.shape)
    if not 1 <= n_components <= max_components:
        raise DataValidationError(
            f"n_components={n_components} inválido; máximo possível é {max_components}."
        )

    matrix = clean.to_numpy(dtype="float64")
    matrix = matrix - matrix.mean(axis=0)  # centralização sempre
    if standardize:
        std = matrix.std(axis=0, ddof=1)
        if np.any(std == 0):
            zero_cols = clean.columns[std == 0].tolist()
            raise DataValidationError(f"Desvio-padrão zero em {zero_cols}; PCA indefinido.")
        matrix = matrix / std

    model = PCA(n_components=n_components)
    scores = model.fit_transform(matrix)

    component_names = [f"PC{i + 1}" for i in range(n_components)]
    result = PcaResult(
        explained_variance_ratio=pd.Series(
            model.explained_variance_ratio_, index=component_names, name="explained_variance_ratio"
        ),
        loadings=pd.DataFrame(model.components_.T, index=clean.columns, columns=component_names),
        factor_returns=pd.DataFrame(scores, index=clean.index, columns=component_names),
    )
    log.info(
        "PCA: %d componentes explicam %.1f%% da variância.",
        n_components,
        100 * float(result.explained_variance_ratio.sum()),
    )
    return result
