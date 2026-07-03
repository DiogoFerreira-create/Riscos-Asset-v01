"""Aquisição de preços e construção de retornos.

Responsabilidades deste módulo (e apenas destas — SRP):

1. **Obter preços e volumes** — :func:`download_market_data`, com três
   fontes possíveis (``yfinance``, sintética, ou automática com
   fallback). O gerador sintético usa Movimento Browniano Geométrico
   (GBM) com semente fixa, permitindo rodar o pipeline inteiro offline
   e escrever testes determinísticos.
2. **Transformar preços em retornos logarítmicos** — :func:`log_returns`.
3. **Tratar outliers por winsorização** — :func:`winsorize_returns`.

Retorno logarítmico é a convenção adotada em todo o projeto porque é
aditivo no tempo (a soma de logs diários é o log do período) — propriedade
que simplifica agregação e modelagem GARCH.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataSource, TRADING_DAYS_PER_YEAR
from .exceptions import DataSourceError, DataValidationError
from .logging_setup import get_logger
from .validation import require_positive

log = get_logger(__name__)


@dataclass(frozen=True)
class MarketData:
    """Par preço/volume alinhado por data e ticker.

    Attributes
    ----------
    prices:
        Preços de fechamento ajustados (linhas = datas, colunas = tickers).
    volumes:
        Volume financeiro diário negociado, mesmas dimensões de ``prices``.
    source:
        Fonte efetivamente utilizada (relevante quando ``AUTO`` faz fallback).
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    source: DataSource


# ---------------------------------------------------------------------------
# Fontes de dados
# ---------------------------------------------------------------------------


def _download_yfinance(tickers: tuple[str, ...], start: str, end: str | None) -> MarketData:
    """Baixa preços ajustados e volumes via ``yfinance``.

    Import local: ``yfinance`` só é exigido se esta fonte for usada,
    permitindo ambientes offline instalarem menos dependências.
    """
    import yfinance as yf  # import tardio proposital

    log.info("Baixando %d tickers via yfinance: %s", len(tickers), ", ".join(tickers))
    raw = yf.download(
        list(tickers), start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )
    if raw is None or raw.empty:
        raise DataSourceError("yfinance retornou conjunto vazio (rede? tickers inválidos?).")

    # yfinance retorna MultiIndex de colunas para >1 ticker e simples para 1.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        volumes = raw["Volume"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
        volumes = raw[["Volume"]].copy()
        volumes.columns = [tickers[0]]

    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise DataSourceError(f"Sem dados para os tickers: {missing}.")

    prices = prices[list(tickers)].dropna(how="all")
    volumes = volumes[list(tickers)].reindex(prices.index)
    # Volume *financeiro* (R$) = quantidade x preço: é o que ADV/DTL consomem.
    volumes = volumes * prices
    return MarketData(prices=prices, volumes=volumes, source=DataSource.YFINANCE)


def _generate_synthetic(
    tickers: tuple[str, ...], start: str, end: str | None, seed: int
) -> MarketData:
    """Gera preços GBM e volumes lognormais, reprodutíveis por semente.

    Modelo por ticker ``i``::

        S_t = S_{t-1} * exp((mu_i - sigma_i^2/2) * dt + sigma_i * sqrt(dt) * Z_t)

    com ``dt = 1/252``, drift/vol anuais sorteados uma vez por ticker a
    partir da semente. Não é um simulador de mercado realista (não há
    caudas pesadas nem clusters de vol); é um *fixture* determinístico
    para desenvolvimento offline e testes — e isso é dito no log.
    """
    log.warning(
        "Usando dados SINTÉTICOS (seed=%d) — apropriado para teste/desenvolvimento, "
        "não para monitoramento de risco real.",
        seed,
    )
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start, end=end_ts)
    if len(dates) < 2:
        raise DataValidationError(f"Período sintético muito curto: {start} a {end_ts.date()}.")

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    prices: dict[str, np.ndarray] = {}
    volumes: dict[str, np.ndarray] = {}
    for ticker in tickers:
        mu = rng.uniform(0.02, 0.15)  # drift anual entre 2% e 15%
        sigma = rng.uniform(0.15, 0.45)  # vol anual entre 15% e 45%
        shocks = rng.standard_normal(len(dates) - 1)
        log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
        start_price = rng.uniform(10.0, 100.0)
        path = start_price * np.exp(np.concatenate([[0.0], np.cumsum(log_increments)]))
        prices[ticker] = path
        # Volume financeiro lognormal ao redor de uma média própria do ticker.
        base_volume = rng.uniform(5e6, 5e7)
        volumes[ticker] = base_volume * np.exp(rng.normal(0.0, 0.3, size=len(dates)))

    prices_df = pd.DataFrame(prices, index=dates)
    volumes_df = pd.DataFrame(volumes, index=dates)
    return MarketData(prices=prices_df, volumes=volumes_df, source=DataSource.SYNTHETIC)


def download_market_data(
    tickers: tuple[str, ...],
    start: str,
    end: str | None = None,
    source: DataSource = DataSource.AUTO,
    synthetic_seed: int = 42,
) -> MarketData:
    """Obtém preços/volumes da fonte configurada.

    Parameters
    ----------
    tickers:
        Códigos dos ativos (padrão Yahoo, ex.: ``"PETR4.SA"``).
    start, end:
        Janela de datas (``end=None`` = até hoje).
    source:
        ``YFINANCE`` (só real), ``SYNTHETIC`` (só simulado) ou ``AUTO``
        (tenta real; sem rede, cai para simulado com ``WARNING`` no log).
    synthetic_seed:
        Semente do gerador sintético (reprodutibilidade).

    Returns
    -------
    MarketData
        Preços e volumes alinhados + fonte efetiva.

    Raises
    ------
    DataSourceError
        Se a fonte exigida falhar (``YFINANCE`` sem rede, por exemplo).
    DataValidationError
        Se a janela de datas for degenerada.

    Examples
    --------
    >>> md = download_market_data(("A", "B"), "2022-01-03", "2022-06-30",
    ...                           source=DataSource.SYNTHETIC, synthetic_seed=7)
    >>> list(md.prices.columns)
    ['A', 'B']
    """
    if not tickers:
        raise DataValidationError("Nenhum ticker informado.")

    if source is DataSource.SYNTHETIC:
        return _generate_synthetic(tickers, start, end, synthetic_seed)

    try:
        return _download_yfinance(tickers, start, end)
    except Exception as exc:  # rede fora, DNS bloqueado, ticker inválido...
        if source is DataSource.YFINANCE:
            raise DataSourceError(f"Falha no yfinance e fallback desabilitado: {exc}") from exc
        log.warning("yfinance indisponível (%s); alternando para dados sintéticos.", exc)
        return _generate_synthetic(tickers, start, end, synthetic_seed)


# ---------------------------------------------------------------------------
# Transformações
# ---------------------------------------------------------------------------


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos logarítmicos ``r_t = ln(P_t / P_{t-1})``.

    Parameters
    ----------
    prices:
        Preços (datas x tickers), estritamente positivos.

    Returns
    -------
    pandas.DataFrame
        Retornos com a primeira linha (sem defasagem disponível) removida.

    Raises
    ------
    DataValidationError
        Se houver preço não positivo (log indefinido) ou menos de duas
        observações por coluna.
    """
    if prices.empty or len(prices) < 2:
        raise DataValidationError("São necessárias >= 2 datas de preço para calcular retornos.")
    if (prices <= 0).any().any():
        offenders = prices.columns[(prices <= 0).any()].tolist()
        raise DataValidationError(f"Preços não positivos em {offenders}; log-retorno indefinido.")
    returns = np.log(prices).diff().dropna(how="all")
    return returns


def winsorize_returns(
    returns: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    """Winsoriza cada coluna nos percentis ``[lower, upper]``.

    Winsorizar (truncar valores extremos no valor do percentil, em vez de
    removê-los) limita a influência de erros de dado e saltos atípicos na
    estimação de covariâncias e do GARCH, preservando o tamanho da amostra.
    O custo é conhecido: caudas verdadeiras são artificialmente encurtadas —
    por isso as métricas de cauda (VaR/CVaR) do pipeline usam os retornos
    *brutos*, e a versão winsorizada alimenta apenas correlação/PCA/GARCH.

    Parameters
    ----------
    returns:
        Retornos (datas x tickers).
    lower, upper:
        Percentis de corte, ``0 <= lower < upper <= 1``.

    Returns
    -------
    pandas.DataFrame
        Cópia winsorizada (NaN preservados onde já existiam).

    Raises
    ------
    DataValidationError
        Se os percentis forem inconsistentes ou o DataFrame estiver vazio.
    """
    if returns.empty:
        raise DataValidationError("DataFrame de retornos vazio.")
    if not 0.0 <= lower < upper <= 1.0:
        raise DataValidationError(f"Percentis inválidos: lower={lower}, upper={upper}.")
    lo = returns.quantile(lower)
    hi = returns.quantile(upper)
    # clip por coluna: alinhamento automático de pandas via axis=1
    return returns.clip(lower=lo, upper=hi, axis=1)


def annualized_volatility(returns: pd.Series) -> float:
    """Volatilidade anualizada: desvio-padrão amostral x sqrt(252).

    Utilitário de conveniência para relatórios; convenção de 252 dias
    úteis definida em :data:`risk_models.config.TRADING_DAYS_PER_YEAR`.
    """
    require_positive(len(returns), "número de observações")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
