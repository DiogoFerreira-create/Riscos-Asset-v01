"""Métricas de liquidez de portfólio: ADV, DTL e slippage bid-ask.

Contexto de negócio
-------------------
Risco de liquidez é a possibilidade de não conseguir desmontar posições
no prazo do passivo (resgates de cotistas) sem impacto relevante de
preço. As três métricas deste módulo respondem, em ordem:

1. **ADV** — quanto o mercado negocia por dia neste papel?
2. **DTL** — quantos dias eu levaria para sair, respeitando um teto de
   participação no volume (para não "ser o mercado")?
3. **Slippage bid-ask** — quanto custa atravessar o spread para executar?
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from .exceptions import DataValidationError
from .logging_setup import get_logger
from .validation import require_positive

log = get_logger(__name__)


class AdvMethod(str, Enum):
    """Estimador do ADV.

    ``MEDIAN`` é o padrão do projeto: a mediana é robusta a dias
    atípicos (leilões, rebalanceamento de índice, vencimento de opções)
    que inflam a média e fariam o DTL parecer melhor do que é —
    conservadorismo correto para um relatório de risco.
    """

    MEAN = "mean"
    MEDIAN = "median"


def average_daily_volume(
    volumes: pd.DataFrame,
    window: int = 21,
    method: AdvMethod = AdvMethod.MEDIAN,
) -> pd.Series:
    """ADV por ativo sobre a janela mais recente.

    Parameters
    ----------
    volumes:
        Volume financeiro diário (datas x tickers), em moeda.
    window:
        Nº de pregões da janela (padrão 21 ~ 1 mês útil).
    method:
        ``MEDIAN`` (robusta, padrão) ou ``MEAN``.

    Returns
    -------
    pandas.Series
        ADV por ticker (moeda/dia), calculado sobre os últimos
        ``window`` pregões disponíveis de cada ativo.

    Raises
    ------
    DataValidationError
        Janela inválida, DataFrame vazio ou histórico menor que a janela.

    Examples
    --------
    >>> vols = pd.DataFrame({"X": [1e6] * 30})
    >>> float(average_daily_volume(vols, window=21)["X"])
    1000000.0
    """
    require_positive(window, "window")
    if volumes.empty:
        raise DataValidationError("DataFrame de volumes vazio.")
    if len(volumes) < window:
        raise DataValidationError(
            f"Histórico de volume ({len(volumes)} dias) menor que a janela ({window})."
        )
    recent = volumes.tail(window)
    if (recent.fillna(0) < 0).any().any():
        raise DataValidationError("Volume negativo encontrado — dado corrompido.")
    adv = recent.median() if method is AdvMethod.MEDIAN else recent.mean()
    return adv.rename("adv")


def days_to_liquidate(
    position_value: pd.Series | dict[str, float],
    adv: pd.Series,
    participation_rate: float = 0.20,
) -> pd.Series:
    """Dias necessários para zerar cada posição.

    Fórmula::

        DTL_i = ceil( posição_i / (participação * ADV_i) )

    A intuição: se posso consumir no máximo ``participação`` (ex.: 20%)
    do volume diário sem mover preço, minha "vazão" de saída diária é
    ``participação * ADV``; o DTL é a posição dividida por essa vazão,
    arredondada para cima (não existe vender por meio dia).

    Parameters
    ----------
    position_value:
        Valor financeiro da posição por ticker (mesma moeda do ADV).
    adv:
        ADV por ticker (saída de :func:`average_daily_volume`).
    participation_rate:
        Fração do ADV consumível por dia, em (0, 1].

    Returns
    -------
    pandas.Series
        DTL (dias inteiros, ``float`` para acomodar ``inf``) por ticker.
        ADV igual a zero resulta em ``inf`` — posição tecnicamente
        ilíquida, sinalizada em log.

    Raises
    ------
    DataValidationError
        Tickers sem ADV correspondente, posição negativa ou taxa de
        participação fora de (0, 1].
    """
    if not 0.0 < participation_rate <= 1.0:
        raise DataValidationError(f"participation_rate={participation_rate} fora de (0, 1].")
    positions = pd.Series(position_value, dtype="float64")
    if (positions < 0).any():
        raise DataValidationError("Posições negativas não suportadas neste cálculo (long-only).")
    missing = positions.index.difference(adv.index)
    if len(missing) > 0:
        raise DataValidationError(f"ADV ausente para: {sorted(map(str, missing))}.")

    capacity = participation_rate * adv.reindex(positions.index)
    with np.errstate(divide="ignore"):  # divisão por ADV=0 vira inf, tratada abaixo
        raw_days = positions / capacity
    dtl = np.ceil(raw_days.replace([np.inf, -np.inf], np.inf))
    dtl = dtl.where(positions > 0, other=0.0)  # posição zero liquida em 0 dias

    illiquid = dtl[np.isinf(dtl)].index.tolist()
    if illiquid:
        log.warning("ADV nulo => DTL infinito para: %s", illiquid)
    return dtl.rename("dtl_days")


def bid_ask_slippage(
    position_value: pd.Series | dict[str, float],
    spread_bps: float,
) -> pd.Series:
    """Custo estimado de atravessar o spread bid-ask ao liquidar.

    Modelo (metade do spread sobre o valor negociado)::

        custo_i = posição_i * (spread_bps / 10_000) / 2

    Racional: o preço "justo" é o mid; quem precisa executar compra no
    ask ou vende no bid, pagando meio spread por transação. É um piso de
    custo — impacto de mercado além do spread não é modelado aqui
    (limitação documentada em ``MODEL_RISK.md``).

    Parameters
    ----------
    position_value:
        Valor financeiro por ticker.
    spread_bps:
        Spread total em pontos-base (1 bp = 0,01%). Deve ser >= 0.

    Returns
    -------
    pandas.Series
        Custo em moeda por ticker.

    Raises
    ------
    DataValidationError
        ``spread_bps`` negativo ou posição negativa.

    Examples
    --------
    >>> costs = bid_ask_slippage({"X": 1_000_000.0}, spread_bps=10)
    >>> float(costs["X"])  # 10 bps => 5 bps de meio-spread => R$ 500
    500.0
    """
    if spread_bps < 0:
        raise DataValidationError(f"spread_bps={spread_bps} não pode ser negativo.")
    positions = pd.Series(position_value, dtype="float64")
    if (positions < 0).any():
        raise DataValidationError("Posições negativas não suportadas neste cálculo.")
    half_spread_fraction = (spread_bps / 10_000.0) / 2.0
    return (positions * half_spread_fraction).rename("slippage_cost")


def liquidity_report(
    volumes: pd.DataFrame,
    weights: tuple[float, ...],
    portfolio_value: float,
    window: int = 21,
    participation_rate: float = 0.20,
    spread_bps: float = 10.0,
) -> pd.DataFrame:
    """Tabela consolidada de liquidez por ativo.

    Distribui ``portfolio_value`` pelos pesos para obter a posição
    financeira por papel e aplica ADV -> DTL -> slippage.

    Returns
    -------
    pandas.DataFrame
        Índice = ticker; colunas ``position_value``, ``adv``,
        ``dtl_days``, ``slippage_cost``.
    """
    require_positive(portfolio_value, "portfolio_value")
    if len(weights) != volumes.shape[1]:
        raise DataValidationError(
            f"{len(weights)} pesos para {volumes.shape[1]} colunas de volume."
        )
    positions = pd.Series(
        {ticker: w * portfolio_value for ticker, w in zip(volumes.columns, weights)},
        name="position_value",
    )
    adv = average_daily_volume(volumes, window=window)
    dtl = days_to_liquidate(positions, adv, participation_rate)
    slip = bid_ask_slippage(positions, spread_bps)
    report = pd.concat([positions, adv, dtl, slip], axis=1)
    log.info("Relatório de liquidez gerado para %d ativos.", len(report))
    return report
