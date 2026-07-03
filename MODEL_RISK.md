"""Validadores de entrada compartilhados pelos módulos de modelo.

Por que um módulo separado?
---------------------------
Todos os modelos (VaR, GARCH, liquidez, fatores, backtest) repetem os
mesmos pré-requisitos: série não vazia, sem tudo-NaN, tamanho mínimo,
nível de confiança em (0,1)... Centralizar evita divergência sutil entre
módulos (ex.: um aceitar série vazia e outro não) e produz mensagens de
erro uniformes — importante quando o usuário só vê o log do Airflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def as_clean_series(data: pd.Series | np.ndarray | list[float], name: str = "série") -> pd.Series:
    """Converte a entrada em ``pd.Series`` de float sem NaN.

    Aceitar ``ndarray``/listas torna as funções de modelo agnósticas à
    origem do dado (DataFrame do pipeline ou array de um teste unitário).

    Parameters
    ----------
    data:
        Sequência numérica 1-D.
    name:
        Nome usado nas mensagens de erro (ex.: ``"retornos de PETR4"``).

    Returns
    -------
    pandas.Series
        Série ``float64`` com os NaN removidos (índice preservado).

    Raises
    ------
    DataValidationError
        Se a entrada for vazia, virar vazia após remover NaN, ou não for 1-D.
    """
    series = pd.Series(data, dtype="float64") if not isinstance(data, pd.Series) else data.astype("float64")
    if series.ndim != 1:
        raise DataValidationError(f"{name}: esperado dado 1-D, recebido ndim={series.ndim}.")
    cleaned = series.dropna()
    if cleaned.empty:
        raise DataValidationError(f"{name}: vazia após remoção de NaN — nada a calcular.")
    return cleaned


def require_min_length(series: pd.Series, minimum: int, name: str = "série") -> None:
    """Garante tamanho mínimo de amostra.

    Raises
    ------
    DataValidationError
        Se ``len(series) < minimum`` — modelos estimados com poucas
        observações produzem números com aparência de precisão e nenhum
        significado estatístico; melhor abortar com mensagem clara.
    """
    if len(series) < minimum:
        raise DataValidationError(
            f"{name}: {len(series)} observações, mínimo exigido {minimum}."
        )


def require_alpha(alpha: float) -> float:
    """Valida nível de confiança ``alpha`` (ex.: 0.95, 0.99).

    Returns
    -------
    float
        O próprio ``alpha`` (permite uso inline: ``a = require_alpha(a)``).

    Raises
    ------
    DataValidationError
        Se ``alpha`` estiver fora do intervalo aberto (0, 1). Valores
        abaixo de 0.5 são aceitos matematicamente, mas o chamador de
        pipeline valida (0.5, 1) na configuração; aqui mantemos a checagem
        matemática mínima para reuso genérico.
    """
    if not 0.0 < alpha < 1.0:
        raise DataValidationError(f"alpha={alpha} inválido; exigido 0 < alpha < 1.")
    return float(alpha)


def require_positive(value: float, name: str) -> float:
    """Valida escalar estritamente positivo (janelas, preços, quantidades)."""
    if value <= 0:
        raise DataValidationError(f"{name}={value} inválido; exigido valor > 0.")
    return float(value)


def align_frames(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Alinha DataFrames pela interseção de índices (datas comuns).

    Uso típico: preços e volumes vindos de fontes distintas. Operar em
    índices desalinhados é fonte clássica de bug silencioso em finanças
    (NaN propagado ou, pior, deslocamento temporal). A interseção é a
    escolha conservadora: só usamos datas em que *tudo* existe.

    Raises
    ------
    DataValidationError
        Se a interseção de datas for vazia.
    """
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    if len(common) == 0:
        raise DataValidationError("Interseção de datas vazia entre os DataFrames fornecidos.")
    return tuple(frame.loc[common] for frame in frames)
