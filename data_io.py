"""Backtesting de modelos de VaR: Kupiec (POF) e Christoffersen (CC).

Por que backtestar o VaR?
-------------------------
Um VaR 99% bem calibrado deve ser violado em ~1% dos dias — nem muito
mais (modelo subestima risco) nem muito menos (modelo superestima e
"custa" capital/limite à toa). Dois testes clássicos formalizam isso:

* **Kupiec (1995), Proportion of Failures** — testa se a *frequência*
  de violações é compatível com ``1 - alpha`` (cobertura incondicional).
* **Christoffersen (1998)** — adiciona o teste de *independência*: as
  violações não devem vir agrupadas (violação hoje não pode prever
  violação amanhã). Cobertura correta **e** independência = cobertura
  condicional correta (LR_cc = LR_uc + LR_ind).

Ambos são testes de razão de verossimilhança com distribuição assintótica
Qui-quadrado (1 g.l. cada; 2 g.l. no combinado).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import DataValidationError
from .logging_setup import get_logger
from .validation import as_clean_series, require_alpha, require_min_length

log = get_logger(__name__)

#: Piso numérico para probabilidades dentro de logs (evita log(0) = -inf
#: quando uma categoria de transição não ocorre na amostra).
_EPS: float = 1e-12


@dataclass(frozen=True)
class BacktestResult:
    """Resultado de um teste de backtesting de VaR.

    Attributes
    ----------
    test_name:
        ``"kupiec_pof"`` ou ``"christoffersen_cc"``.
    statistic:
        Estatística LR do teste.
    p_value:
        P-valor sob a distribuição Qui-quadrado de referência.
    reject_at_5pct:
        ``True`` se o modelo é rejeitado a 5% de significância
        (p-valor < 0.05) — leia-se "há evidência de má calibração".
    n_observations, n_violations:
        Tamanho da amostra e nº de violações observadas.
    expected_violations:
        Violações esperadas sob calibração perfeita: ``n * (1 - alpha)``.
    """

    test_name: str
    statistic: float
    p_value: float
    reject_at_5pct: bool
    n_observations: int
    n_violations: int
    expected_violations: float


def compute_violations(returns: pd.Series | np.ndarray, var_estimate: float) -> pd.Series:
    """Série binária de violações do VaR (1 = perda excedeu o VaR).

    Violação: ``r_t < -VaR`` (lembrando que o VaR é reportado positivo).

    Parameters
    ----------
    returns:
        Retornos realizados no período de teste.
    var_estimate:
        VaR (positivo, fração) contra o qual comparar. Nesta versão o
        VaR é mantido fixo no período (VaR "estático"); backtest com VaR
        recalculado em janela móvel está no roadmap (README, seção
        *Próximos passos*).

    Returns
    -------
    pandas.Series
        Inteiros 0/1 indexados como os retornos.

    Raises
    ------
    DataValidationError
        Retornos vazios ou ``var_estimate`` não positivo.
    """
    if var_estimate <= 0:
        raise DataValidationError(f"var_estimate={var_estimate} deve ser positivo.")
    series = as_clean_series(returns, "retornos (violações)")
    return (series < -var_estimate).astype(int).rename("violation")


def kupiec_pof_test(violations: pd.Series | np.ndarray, alpha: float) -> BacktestResult:
    """Teste de Proporção de Falhas de Kupiec (cobertura incondicional).

    Hipótese nula: P(violação) = ``p = 1 - alpha``. Estatística::

        LR_uc = -2 * ln[ (1-p)^{n-x} p^x / ((1-p̂)^{n-x} p̂^x) ],  p̂ = x/n

    com ``x`` violações em ``n`` dias; LR_uc ~ Qui²(1) sob H0.

    Parameters
    ----------
    violations:
        Série binária 0/1 (saída de :func:`compute_violations`).
    alpha:
        Nível de confiança do VaR testado (ex.: 0.99).

    Returns
    -------
    BacktestResult

    Raises
    ------
    DataValidationError
        Série vazia, valores fora de {0,1} ou ``alpha`` inválido.

    Notes
    -----
    Casos de borda: com ``x = 0`` ou ``x = n``, o termo do estimador
    p̂ degenera; usamos o piso :data:`_EPS` dentro dos logs, prática
    padrão que mantém a estatística finita e conservadora.
    """
    alpha = require_alpha(alpha)
    v = as_clean_series(violations, "violações").astype(int)
    require_min_length(v, 30, "violações (Kupiec)")
    if not set(np.unique(v)).issubset({0, 1}):
        raise DataValidationError("Série de violações deve conter apenas 0/1.")

    n = int(len(v))
    x = int(v.sum())
    p = 1.0 - alpha
    p_hat = x / n

    log_l0 = (n - x) * np.log(max(1.0 - p, _EPS)) + x * np.log(max(p, _EPS))
    log_l1 = (n - x) * np.log(max(1.0 - p_hat, _EPS)) + x * np.log(max(p_hat, _EPS))
    lr_uc = float(-2.0 * (log_l0 - log_l1))
    p_value = float(stats.chi2.sf(lr_uc, df=1))

    result = BacktestResult(
        test_name="kupiec_pof",
        statistic=lr_uc,
        p_value=p_value,
        reject_at_5pct=p_value < 0.05,
        n_observations=n,
        n_violations=x,
        expected_violations=n * p,
    )
    log.info(
        "Kupiec: %d violações em %d dias (esperado %.1f) | LR=%.3f p=%.4f",
        x, n, n * p, lr_uc, p_value,
    )
    return result


def _independence_lr(v: np.ndarray) -> float:
    """Estatística LR de independência de Christoffersen (Qui², 1 g.l.).

    Conta as transições consecutivas ``n_ij`` (de estado i para j,
    i,j em {0,1}) e compara a verossimilhança de uma cadeia de Markov de
    1ª ordem (probabilidades de violação diferentes após dia calmo vs.
    após dia de violação) contra a de violações i.i.d. Sob H0
    (independência), pi_{0->1} = pi_{1->1}.
    """
    current, nxt = v[:-1], v[1:]
    n00 = int(np.sum((current == 0) & (nxt == 0)))
    n01 = int(np.sum((current == 0) & (nxt == 1)))
    n10 = int(np.sum((current == 1) & (nxt == 0)))
    n11 = int(np.sum((current == 1) & (nxt == 1)))

    pi01 = n01 / max(n00 + n01, 1)  # P(violação | ontem calmo)
    pi11 = n11 / max(n10 + n11, 1)  # P(violação | ontem violação)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)  # incondicional

    log_l0 = (n00 + n10) * np.log(max(1 - pi, _EPS)) + (n01 + n11) * np.log(max(pi, _EPS))
    log_l1 = (
        n00 * np.log(max(1 - pi01, _EPS))
        + n01 * np.log(max(pi01, _EPS))
        + n10 * np.log(max(1 - pi11, _EPS))
        + n11 * np.log(max(pi11, _EPS))
    )
    return float(-2.0 * (log_l0 - log_l1))


def christoffersen_test(violations: pd.Series | np.ndarray, alpha: float) -> BacktestResult:
    """Teste de Cobertura Condicional de Christoffersen.

    Combina cobertura incondicional (Kupiec) e independência::

        LR_cc = LR_uc + LR_ind  ~  Qui²(2) sob H0

    Rejeitar aqui significa que o modelo erra na frequência de violações,
    na sua distribuição temporal (clusters), ou em ambas.

    Parameters
    ----------
    violations:
        Série binária 0/1.
    alpha:
        Nível de confiança do VaR testado.

    Returns
    -------
    BacktestResult
        ``statistic`` = LR_cc; ``p_value`` sob Qui²(2).
    """
    alpha = require_alpha(alpha)
    v_series = as_clean_series(violations, "violações").astype(int)
    require_min_length(v_series, 30, "violações (Christoffersen)")
    if not set(np.unique(v_series)).issubset({0, 1}):
        raise DataValidationError("Série de violações deve conter apenas 0/1.")

    kupiec = kupiec_pof_test(v_series, alpha)
    lr_ind = _independence_lr(v_series.to_numpy())
    lr_cc = kupiec.statistic + lr_ind
    p_value = float(stats.chi2.sf(lr_cc, df=2))

    result = BacktestResult(
        test_name="christoffersen_cc",
        statistic=float(lr_cc),
        p_value=p_value,
        reject_at_5pct=p_value < 0.05,
        n_observations=kupiec.n_observations,
        n_violations=kupiec.n_violations,
        expected_violations=kupiec.expected_violations,
    )
    log.info("Christoffersen: LR_cc=%.3f (LR_ind=%.3f) p=%.4f", lr_cc, lr_ind, p_value)
    return result


def backtest_report(
    returns: pd.Series,
    var_by_alpha: dict[float, float],
) -> pd.DataFrame:
    """Roda Kupiec e Christoffersen para cada nível de VaR fornecido.

    Parameters
    ----------
    returns:
        Retornos realizados do portfólio.
    var_by_alpha:
        Mapa ``alpha -> VaR estimado`` (positivo, fração).

    Returns
    -------
    pandas.DataFrame
        Uma linha por (alpha, teste) com estatística, p-valor e decisão.
    """
    rows: list[dict[str, object]] = []
    for alpha, var_value in sorted(var_by_alpha.items()):
        violations = compute_violations(returns, var_value)
        for test in (kupiec_pof_test(violations, alpha), christoffersen_test(violations, alpha)):
            rows.append(
                {
                    "alpha": alpha,
                    "var_estimate": var_value,
                    "test": test.test_name,
                    "statistic": test.statistic,
                    "p_value": test.p_value,
                    "reject_at_5pct": test.reject_at_5pct,
                    "n_violations": test.n_violations,
                    "expected_violations": test.expected_violations,
                }
            )
    return pd.DataFrame(rows)
