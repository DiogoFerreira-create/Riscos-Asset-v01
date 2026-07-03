"""``risk_models`` — modelos quantitativos de risco de mercado e liquidez.

Este ``__init__`` define a **API pública** do pacote: tudo que o pipeline
(e usuários externos) devem importar está reexportado aqui. Módulos podem
ser reorganizados internamente sem quebrar quem consome o pacote, desde
que estes nomes permaneçam estáveis.

Exemplo de uso rápido::

    from risk_models import (
        download_market_data, log_returns, historical_var, fit_garch,
    )
"""

from .backtest import (
    BacktestResult,
    backtest_report,
    christoffersen_test,
    compute_violations,
    kupiec_pof_test,
)
from .config import (
    DataSource,
    LimitsConfig,
    PathsConfig,
    PipelineParams,
    RiskConfig,
    TRADING_DAYS_PER_YEAR,
    VarMethod,
    load_config,
)
from .data_io import load_parquet, save_parquet
from .exceptions import (
    ConfigurationError,
    DataSourceError,
    DataValidationError,
    ModelFitError,
    RiskModelError,
)
from .factors import PcaResult, correlation_matrix, pca_factors
from .garch import GarchResult, fit_garch, garch_report
from .liquidity import (
    AdvMethod,
    average_daily_volume,
    bid_ask_slippage,
    days_to_liquidate,
    liquidity_report,
)
from .logging_setup import get_logger, setup_logging
from .reporting import (
    build_dashboard_html,
    figure_correlation,
    figure_garch,
    figure_limits_table,
    figure_liquidity,
    figure_normalized_prices,
    figure_pca,
    figure_portfolio_var,
    figure_var_by_asset,
)
from .returns import (
    MarketData,
    annualized_volatility,
    download_market_data,
    log_returns,
    winsorize_returns,
)
from .var import (
    conditional_var,
    historical_var,
    parametric_var,
    portfolio_returns,
    var_report,
)

__version__ = "1.0.0"

__all__ = [
    # config
    "DataSource", "VarMethod", "PathsConfig", "LimitsConfig", "PipelineParams",
    "RiskConfig", "load_config", "TRADING_DAYS_PER_YEAR",
    # exceptions
    "RiskModelError", "ConfigurationError", "DataValidationError",
    "DataSourceError", "ModelFitError",
    # logging / io
    "setup_logging", "get_logger", "save_parquet", "load_parquet",
    # returns
    "MarketData", "download_market_data", "log_returns", "winsorize_returns",
    "annualized_volatility",
    # var
    "historical_var", "parametric_var", "conditional_var",
    "portfolio_returns", "var_report",
    # garch
    "GarchResult", "fit_garch", "garch_report",
    # liquidity
    "AdvMethod", "average_daily_volume", "days_to_liquidate",
    "bid_ask_slippage", "liquidity_report",
    # factors
    "PcaResult", "correlation_matrix", "pca_factors",
    # backtest
    "BacktestResult", "compute_violations", "kupiec_pof_test",
    "christoffersen_test", "backtest_report",
    # reporting
    "build_dashboard_html", "figure_normalized_prices", "figure_portfolio_var",
    "figure_var_by_asset", "figure_garch", "figure_correlation", "figure_pca",
    "figure_liquidity", "figure_limits_table",
]
