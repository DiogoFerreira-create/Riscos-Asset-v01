"""Value-at-Risk e Expected Shortfall (CVaR).

Convenção de sinal (importante!)
--------------------------------
Todas as funções retornam risco como **número positivo em fração do
patrimônio**: ``VaR 99% = 0.032`` significa "com 99% de confiança, a
perda diária não excede 3,2%". Internamente trabalhamos com retornos
(ganhos positivos, perdas negativas) e invertemos o sinal só na saída.
Fixar a convenção em um único lugar elimina a fonte nº 1 de bugs em
código de risco: sinais trocados entre camadas.

Metodologias implementadas
--------------------------
* **Histórica** — quantil empírico da distribuição de retornos; não
  assume forma distribucional, mas "só conhece" o que está na janela.
* **Paramétrica (Delta-Normal)** — assume retornos ~ Normal(mu, sigma²):
  ``VaR_alpha = -(mu + z_{1-alpha} * sigma)``. Rápida e suave, porém
  subestima caudas pesadas (curtose) típicas de retornos financeiros.
* **CVaR / Expected Shortfall** — média das perdas *além* do VaR;
  responde "quando o VaR estoura, quão ruim fica, em média?" e é
  subaditiva (medida coerente de risco, Artzner et al., 1999).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import VarMethod
from .exceptions import DataValidationError
from .logging_setup import get_logger
from .validation import as_clean_series, require_alpha, require_min_length

log = get_logger(__name__)

#: Amostra mínima para estimar um quantil de cauda com o mínimo de dignidade.
#: Com menos de ~60 pontos, o quantil de 99% é definido por 1 observação.
MIN_OBS_VAR: int = 60

#: Piso numérico para considerar um desvio-padrão "zero". Comparar float
#: com ``== 0.0`` falha para séries constantes cujo valor não é exatamente
#: representável em binário (ex.: 0.001): os desvios ficam em ~1e-19, não 0.
#: Retornos diários reais têm sigma >= 1e-4, então 1e-12 separa com folga
#: "constante a menos de ruído de máquina" de qualquer série genuína.
_SIGMA_FLOOR: float = 1e-12


def historical_var(returns: pd.Series | np.ndarray, alpha: float = 0.99) -> float:
    """VaR histórico (não paramétrico) no nível ``alpha``.

    Definição: o quantil ``1 - alpha`` da distribuição empírica dos
    retornos, com sinal invertido. Ex.: alpha=0.99 usa o percentil 1.

    Parameters
    ----------
    returns:
        Série de retornos (fração, não %). NaN são descartados.
    alpha:
        Nível de confiança em (0, 1); padrão 0.99.

    Returns
    -------
    float
        VaR como perda positiva (fração do patrimônio).

    Raises
    ------
    DataValidationError
        Série vazia/curta demais ou ``alpha`` fora de (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> r = np.concatenate([np.full(99, 0.001), [-0.05]])
    >>> round(historical_var(r, alpha=0.99), 4)
    0.0449
    """
    alpha = require_alpha(alpha)
    series = as_clean_series(returns, "retornos")
    require_min_length(series, MIN_OBS_VAR, "retornos (VaR histórico)")
    # interpolation="linear" (padrão) evita depender de 1 único ponto na cauda.
    quantile = float(series.quantile(1.0 - alpha))
    return -quantile


def parametric_var(returns: pd.Series | np.ndarray, alpha: float = 0.99) -> float:
    """VaR paramétrico Delta-Normal no nível ``alpha``.

    Fórmula: ``VaR = -(mu + z_{1-alpha} * sigma)``, com ``z`` da Normal
    padrão. Note que ``z_{1-alpha}`` é negativo (ex.: -2.326 para 99%),
    logo o VaR resulta positivo quando ``sigma`` domina ``mu`` — o caso
    típico em horizonte diário.

    Parameters
    ----------
    returns:
        Série de retornos diários.
    alpha:
        Nível de confiança em (0, 1).

    Returns
    -------
    float
        VaR como perda positiva (fração do patrimônio).

    Raises
    ------
    DataValidationError
        Série curta/vazia, ``alpha`` inválido ou desvio-padrão nulo
        (série constante não permite estimar risco).
    """
    alpha = require_alpha(alpha)
    series = as_clean_series(returns, "retornos")
    require_min_length(series, MIN_OBS_VAR, "retornos (VaR paramétrico)")
    mu = float(series.mean())
    sigma = float(series.std(ddof=1))
    if sigma < _SIGMA_FLOOR:
        raise DataValidationError("Desvio-padrão zero: série constante não tem VaR definível.")
    z = float(stats.norm.ppf(1.0 - alpha))
    return -(mu + z * sigma)


def conditional_var(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.99,
    method: VarMethod = VarMethod.HISTORICAL,
) -> float:
    """CVaR / Expected Shortfall no nível ``alpha``.

    * ``HISTORICAL``: média dos retornos abaixo do quantil ``1 - alpha``
      (média empírica da cauda), com sinal invertido.
    * ``PARAMETRIC``: fórmula fechada sob Normalidade:
      ``ES = -(mu - sigma * phi(z_{1-alpha}) / (1 - alpha))``,
      onde ``phi`` é a densidade Normal padrão.

    Parameters
    ----------
    returns:
        Série de retornos diários.
    alpha:
        Nível de confiança em (0, 1).
    method:
        :class:`~risk_models.config.VarMethod` — casa com a metodologia
        de VaR usada, para comparabilidade VaR vs. CVaR.

    Returns
    -------
    float
        Expected Shortfall como perda positiva. Sempre ``>= VaR`` do
        mesmo método/nível (propriedade verificada em teste unitário).

    Raises
    ------
    DataValidationError
        Entradas inválidas (ver validadores) ou cauda empírica vazia.
    """
    alpha = require_alpha(alpha)
    series = as_clean_series(returns, "retornos")
    require_min_length(series, MIN_OBS_VAR, "retornos (CVaR)")

    if method is VarMethod.PARAMETRIC:
        mu = float(series.mean())
        sigma = float(series.std(ddof=1))
        if sigma < _SIGMA_FLOOR:
            raise DataValidationError("Desvio-padrão zero: série constante não tem CVaR definível.")
        z = float(stats.norm.ppf(1.0 - alpha))
        es = -(mu - sigma * float(stats.norm.pdf(z)) / (1.0 - alpha))
        return es

    threshold = float(series.quantile(1.0 - alpha))
    tail = series[series <= threshold]
    if tail.empty:
        # Amostra pequena + alpha alto podem esvaziar a cauda estrita;
        # o quantil em si é o pior caso observável.
        log.warning("Cauda vazia para alpha=%.3f; CVaR reduzido ao próprio quantil.", alpha)
        return -threshold
    return -float(tail.mean())


def portfolio_returns(returns: pd.DataFrame, weights: tuple[float, ...] | np.ndarray) -> pd.Series:
    """Retorno do portfólio: combinação linear ``R @ w`` dos retornos.

    Nota metodológica: a combinação linear é exata para retornos
    aritméticos; para log-retornos diários é uma aproximação de primeira
    ordem, excelente em horizonte de 1 dia (erro ~ O(r²)) e padrão de
    mercado para VaR diário. A limitação está documentada em
    ``MODEL_RISK.md``.

    Parameters
    ----------
    returns:
        Retornos por ativo (datas x tickers).
    weights:
        Pesos na ordem das colunas; devem somar 1 (tolerância 1e-6).

    Returns
    -------
    pandas.Series
        Série de retornos do portfólio (linhas com qualquer NaN são
        descartadas para não contaminar a soma ponderada).

    Raises
    ------
    DataValidationError
        Dimensões incompatíveis ou pesos que não somam 1.
    """
    w = np.asarray(weights, dtype="float64")
    if returns.shape[1] != w.shape[0]:
        raise DataValidationError(
            f"{returns.shape[1]} colunas de retorno para {w.shape[0]} pesos."
        )
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise DataValidationError(f"Pesos somam {w.sum():.6f}; esperado 1.0.")
    clean = returns.dropna(how="any")
    if clean.empty:
        raise DataValidationError("Sem datas completas (todas as linhas contêm NaN).")
    return pd.Series(clean.to_numpy() @ w, index=clean.index, name="portfolio")


def var_report(
    returns: pd.DataFrame,
    weights: tuple[float, ...],
    alphas: tuple[float, ...],
    portfolio_value: float,
) -> pd.DataFrame:
    """Tabela consolidada de VaR/CVaR por ativo e do portfólio.

    Para cada ativo e para o portfólio agregado calcula, em cada nível de
    ``alphas``: VaR histórico, VaR paramétrico e CVaR histórico — em
    fração e em moeda (``* portfolio_value``, no caso do portfólio; para
    ativos individuais, em fração apenas, pois a exposição monetária de
    cada um depende do peso).

    Returns
    -------
    pandas.DataFrame
        Índice = nome do ativo (ou ``"PORTFOLIO"``); colunas no formato
        ``var_hist_99``, ``var_param_99``, ``cvar_99`` etc., mais
        ``var_hist_99_brl`` para o portfólio.
    """
    rows: dict[str, dict[str, float]] = {}

    def _metrics(series: pd.Series) -> dict[str, float]:
        out: dict[str, float] = {}
        for alpha in alphas:
            tag = f"{int(round(alpha * 100))}"
            out[f"var_hist_{tag}"] = historical_var(series, alpha)
            out[f"var_param_{tag}"] = parametric_var(series, alpha)
            out[f"cvar_{tag}"] = conditional_var(series, alpha, VarMethod.HISTORICAL)
        return out

    for column in returns.columns:
        rows[str(column)] = _metrics(returns[column])

    port = portfolio_returns(returns, weights)
    port_metrics = _metrics(port)
    # Métricas monetárias apenas no agregado: é onde o PL de referência se aplica.
    for key, value in list(port_metrics.items()):
        port_metrics[f"{key}_brl"] = value * portfolio_value
    rows["PORTFOLIO"] = port_metrics

    report = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    log.info("Relatório de VaR gerado para %d ativos + portfólio.", len(returns.columns))
    return report
