"""Pipeline diário de riscos — DAG do Airflow com fallback local.

Duplo modo de execução
----------------------
* **Com Airflow instalado**: este arquivo define a DAG ``risk_pipeline``
  (agenda ``08:00 America/Recife``, segunda a sexta) com uma
  ``PythonOperator`` por tarefa, encadeadas linearmente.
* **Sem Airflow** (``python dags/risk_pipeline.py``): a função
  :func:`run_local_pipeline` executa as mesmas tarefas em sequência —
  útil para desenvolvimento, depuração e CI.

Contrato entre tarefas
----------------------
As tarefas **não** trocam dados por XCom: cada uma lê os Parquet da(s)
anterior(es) em ``data/curated_risk_pipeline/`` e grava o seu. Isso as
torna idempotentes e individualmente re-executáveis, e deixa um rastro
auditável em disco (ver justificativa em ``risk_models.data_io``).

Artefatos produzidos (nomes em ``ARTIFACTS``)::

    prices.parquet, volumes.parquet          (task_extract_prices)
    returns.parquet, returns_winsor.parquet  (task_compute_returns)
    var_report.parquet                       (task_var_metrics)
    garch_report.parquet                     (task_garch_vol)
    liquidity_report.parquet                 (task_liquidity_metrics)
    correlation.parquet, pca_*.parquet       (task_factor_analysis)
    backtest_report.parquet                  (task_var_backtest)
    outputs/tables/risk_summary.csv          (task_consolidate)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap de import: quando executado direto (ou copiado para a pasta de
# DAGs do Airflow), o pacote ``risk_models`` em ``src/`` não está no path.
# Inserimos <raiz>/src explicitamente — alternativa robusta a depender de
# ``pip install -e .`` ter sido executado no ambiente do scheduler.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402  (após bootstrap de path, proposital)

from risk_models import (  # noqa: E402
    RiskConfig,
    backtest_report,
    build_dashboard_html,
    correlation_matrix,
    download_market_data,
    figure_correlation,
    figure_garch,
    figure_limits_table,
    figure_liquidity,
    figure_normalized_prices,
    figure_pca,
    figure_portfolio_var,
    figure_var_by_asset,
    garch_report,
    get_logger,
    historical_var,
    liquidity_report,
    load_config,
    load_parquet,
    log_returns,
    pca_factors,
    portfolio_returns,
    save_parquet,
    setup_logging,
    var_report,
    winsorize_returns,
)

log = get_logger(__name__)

#: Nomes canônicos dos artefatos intermediários (chave única -> arquivo).
ARTIFACTS: dict[str, str] = {
    "prices": "prices.parquet",
    "volumes": "volumes.parquet",
    "returns": "returns.parquet",
    "returns_winsor": "returns_winsor.parquet",
    "var_report": "var_report.parquet",
    "garch_report": "garch_report.parquet",
    "liquidity_report": "liquidity_report.parquet",
    "correlation": "correlation.parquet",
    "pca_variance": "pca_explained_variance.parquet",
    "pca_loadings": "pca_loadings.parquet",
    "backtest_report": "backtest_report.parquet",
}


def _artifact(cfg: RiskConfig, key: str) -> Path:
    """Caminho completo do artefato ``key`` na pasta curada."""
    return Path(cfg.paths.curated_dir) / ARTIFACTS[key]


# ---------------------------------------------------------------------------
# Tarefas (cada função = 1 nó da DAG; assinatura sem argumentos além de cfg
# para que o PythonOperator as chame uniformemente)
# ---------------------------------------------------------------------------


def task_extract_prices(cfg: RiskConfig) -> None:
    """Extrai preços/volumes da fonte configurada e persiste em Parquet."""
    p = cfg.params
    market = download_market_data(
        tickers=p.tickers,
        start=p.start_date,
        end=p.end_date,
        source=p.data_source,
        synthetic_seed=p.synthetic_seed,
    )
    if len(market.prices) < cfg.limits.min_history_days:
        log.warning(
            "Histórico (%d dias) abaixo do mínimo configurado (%d): "
            "modelos seguirão, mas com menor confiabilidade estatística.",
            len(market.prices), cfg.limits.min_history_days,
        )
    save_parquet(market.prices, _artifact(cfg, "prices"))
    save_parquet(market.volumes, _artifact(cfg, "volumes"))


def task_compute_returns(cfg: RiskConfig) -> None:
    """Preços -> log-retornos brutos + versão winsorizada."""
    prices = load_parquet(_artifact(cfg, "prices"))
    returns = log_returns(prices)
    winsorized = winsorize_returns(returns, cfg.params.winsor_lower, cfg.params.winsor_upper)
    save_parquet(returns, _artifact(cfg, "returns"))
    save_parquet(winsorized, _artifact(cfg, "returns_winsor"))


def task_var_metrics(cfg: RiskConfig) -> None:
    """VaR/CVaR por ativo e do portfólio (retornos BRUTOS: cauda intacta)."""
    returns = load_parquet(_artifact(cfg, "returns"))
    report = var_report(
        returns,
        weights=cfg.params.weights,
        alphas=cfg.params.var_alphas,
        portfolio_value=cfg.params.portfolio_value,
    )
    save_parquet(report, _artifact(cfg, "var_report"))


def task_garch_vol(cfg: RiskConfig) -> None:
    """GARCH(1,1) por ativo (retornos WINSORIZADOS: estimação estável)."""
    returns = load_parquet(_artifact(cfg, "returns_winsor"))
    report = garch_report(returns)
    high_persistence = report[report["persistence"] >= cfg.limits.garch_persistence_max]
    if not high_persistence.empty:
        log.warning(
            "Persistência GARCH >= %.3f em: %s",
            cfg.limits.garch_persistence_max,
            ", ".join(high_persistence.index.astype(str)),
        )
    save_parquet(report, _artifact(cfg, "garch_report"))


def task_liquidity_metrics(cfg: RiskConfig) -> None:
    """ADV, DTL e slippage a partir do volume financeiro real da extração."""
    volumes = load_parquet(_artifact(cfg, "volumes"))
    report = liquidity_report(
        volumes,
        weights=cfg.params.weights,
        portfolio_value=cfg.params.portfolio_value,
        window=cfg.params.adv_window,
        participation_rate=cfg.params.participation_rate,
        spread_bps=cfg.params.spread_bps,
    )
    save_parquet(report, _artifact(cfg, "liquidity_report"))


def task_factor_analysis(cfg: RiskConfig) -> None:
    """Correlação (janela móvel) + PCA sobre retornos winsorizados."""
    returns = load_parquet(_artifact(cfg, "returns_winsor"))
    corr = correlation_matrix(returns, window=cfg.params.corr_window)
    save_parquet(corr, _artifact(cfg, "correlation"))

    pca = pca_factors(returns, n_components=min(cfg.params.pca_components, returns.shape[1]))
    save_parquet(pca.explained_variance_ratio.to_frame(), _artifact(cfg, "pca_variance"))
    save_parquet(pca.loadings, _artifact(cfg, "pca_loadings"))


def task_var_backtest(cfg: RiskConfig) -> None:
    """Kupiec + Christoffersen sobre o VaR histórico do portfólio."""
    returns = load_parquet(_artifact(cfg, "returns"))
    port = portfolio_returns(returns, cfg.params.weights)
    var_by_alpha = {alpha: historical_var(port, alpha) for alpha in cfg.params.var_alphas}
    report = backtest_report(port, var_by_alpha)
    save_parquet(report, _artifact(cfg, "backtest_report"))


def task_consolidate(cfg: RiskConfig) -> None:
    """Consolida um resumo executivo (CSV) e checa limites de risco.

    O resumo cruza VaR/CVaR do portfólio e DTL máximo com os limites de
    ``configs/limits.yaml``; violações viram ``WARNING`` no log e coluna
    ``breach`` no arquivo — visível para quem só abre o CSV.
    """
    var_rep = load_parquet(_artifact(cfg, "var_report"))
    liq_rep = load_parquet(_artifact(cfg, "liquidity_report"))

    port = var_rep.loc["PORTFOLIO"]
    finite_dtl = liq_rep["dtl_days"].replace([float("inf")], pd.NA).dropna()
    worst_dtl = float(finite_dtl.max()) if not finite_dtl.empty else float("inf")

    checks = pd.DataFrame(
        [
            {
                "metric": "portfolio_var_hist_99",
                "value": float(port["var_hist_99"]),
                "limit": cfg.limits.var_99_max,
                "breach": bool(port["var_hist_99"] > cfg.limits.var_99_max),
            },
            {
                "metric": "portfolio_cvar_99",
                "value": float(port["cvar_99"]),
                "limit": cfg.limits.cvar_99_max,
                "breach": bool(port["cvar_99"] > cfg.limits.cvar_99_max),
            },
            {
                "metric": "max_dtl_days",
                "value": worst_dtl,
                "limit": cfg.limits.dtl_max_days,
                "breach": bool(worst_dtl > cfg.limits.dtl_max_days),
            },
        ]
    )
    for _, row in checks[checks["breach"]].iterrows():
        log.warning("LIMITE VIOLADO: %s = %.4f > %.4f", row["metric"], row["value"], row["limit"])

    tables_dir = Path(cfg.paths.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "risk_summary.csv"
    checks.to_csv(summary_path, index=False)
    log.info("Resumo executivo gravado em %s", summary_path)


def task_generate_dashboard(cfg: RiskConfig) -> None:
    """Monta o dashboard HTML interativo a partir dos artefatos do dia.

    Cada seção recebe um *construtor* (lambda sem argumentos): se um
    artefato faltar ou um gráfico falhar, a seção vira aviso no HTML e
    as demais são geradas — o comitê recebe o dashboard possível, nunca
    dashboard nenhum.
    """
    prices = load_parquet(_artifact(cfg, "prices"))
    returns = load_parquet(_artifact(cfg, "returns"))
    var_rep = load_parquet(_artifact(cfg, "var_report"))
    garch_rep = load_parquet(_artifact(cfg, "garch_report"))
    corr = load_parquet(_artifact(cfg, "correlation"))
    pca_var = load_parquet(_artifact(cfg, "pca_variance")).iloc[:, 0]
    pca_load = load_parquet(_artifact(cfg, "pca_loadings"))
    liq_rep = load_parquet(_artifact(cfg, "liquidity_report"))

    port = portfolio_returns(returns, cfg.params.weights)
    var_by_alpha = {a: historical_var(port, a) for a in cfg.params.var_alphas}
    summary_path = Path(cfg.paths.tables_dir) / "risk_summary.csv"
    summary = pd.read_csv(summary_path)

    # Rótulo honesto da fonte no cabeçalho: sintético precisa estar visível.
    source = str(cfg.params.data_source.value)

    figures = [
        ("Checagem de limites", lambda: figure_limits_table(summary)),
        ("Preços normalizados", lambda: figure_normalized_prices(prices)),
        ("Portfólio: VaR e violações",
         lambda: figure_portfolio_var(port, var_by_alpha)),
        ("VaR/CVaR por ativo", lambda: figure_var_by_asset(var_rep)),
        ("GARCH", lambda: figure_garch(garch_rep)),
        ("Correlação", lambda: figure_correlation(corr)),
        ("PCA", lambda: figure_pca(pca_var, pca_load)),
        ("Liquidez (DTL)",
         lambda: figure_liquidity(liq_rep, dtl_limit=cfg.limits.dtl_max_days)),
    ]
    output = Path(cfg.paths.reports_dir) / "risk_dashboard.html"
    build_dashboard_html(figures, output, source_label=source)


#: Ordem canônica de execução — única fonte de verdade usada tanto pelo
#: modo local quanto pela montagem da DAG (evita divergência entre os dois).
PIPELINE_TASKS = (
    task_extract_prices,
    task_compute_returns,
    task_var_metrics,
    task_garch_vol,
    task_liquidity_metrics,
    task_factor_analysis,
    task_var_backtest,
    task_consolidate,
    task_generate_dashboard,
)


def run_local_pipeline(config_dir: Path | str | None = None) -> None:
    """Executa todas as tarefas em sequência, fora do Airflow.

    Parameters
    ----------
    config_dir:
        Pasta alternativa de configs (testes usam uma temporária).
    """
    cfg = load_config(config_dir)
    cfg.paths.ensure_directories()
    setup_logging(cfg.paths.logs_dir)
    log.info("=== Pipeline local iniciado (%d tarefas) ===", len(PIPELINE_TASKS))
    for task in PIPELINE_TASKS:
        log.info("--- %s ---", task.__name__)
        task(cfg)
    log.info("=== Pipeline local concluído com sucesso ===")


# ---------------------------------------------------------------------------
# Definição da DAG (somente se o Airflow estiver disponível no ambiente)
# ---------------------------------------------------------------------------
try:  # pragma: no cover — exercitado apenas em ambiente com Airflow
    import pendulum
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _cfg = load_config()
    _cfg.paths.ensure_directories()
    setup_logging(_cfg.paths.logs_dir)

    with DAG(
        dag_id="risk_pipeline",
        description="Cálculo diário de métricas de risco de mercado e liquidez.",
        schedule="0 8 * * 1-5",  # 08:00, segunda a sexta
        start_date=pendulum.datetime(2025, 1, 1, tz="America/Recife"),
        catchup=False,  # não reprocessa dias passados automaticamente
        default_args={"retries": 2},
        tags=["risco", "quant"],
    ) as dag:
        operators = [
            PythonOperator(
                task_id=task.__name__,
                python_callable=task,
                op_kwargs={"cfg": _cfg},
            )
            for task in PIPELINE_TASKS
        ]
        # Encadeamento linear: t0 >> t1 >> ... >> tN
        for upstream, downstream in zip(operators, operators[1:]):
            upstream >> downstream

except ImportError:
    # Airflow ausente: modo local disponível via execução direta.
    if __name__ == "__main__":
        run_local_pipeline()
