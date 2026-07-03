"""Camada de configuração tipada do projeto.

Toda a parametrização vive em três arquivos YAML dentro de ``configs/``:

* ``paths.yaml``    — diretórios de dados/saídas/logs;
* ``limits.yaml``   — limites de risco monitorados pelo pipeline;
* ``pipeline.yaml`` — universo de ativos, pesos, janelas e parâmetros.

Regras de precedência (da maior para a menor):

1. **Variável de ambiente** (ex.: ``RISK_CURATED_DIR``) — permite que o
   mesmo código rode em dev/produção sem editar arquivos;
2. **Valor do YAML**;
3. **Padrão embutido** nos ``dataclasses`` abaixo.

O carregamento é validado: valores impossíveis (alpha fora de (0,1),
pesos que não somam 1, janelas não positivas) falham cedo com
:class:`~risk_models.exceptions.ConfigurationError`, em vez de produzir
números silenciosamente errados no meio do pipeline.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

from .exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Constantes de projeto
# ---------------------------------------------------------------------------

#: Raiz do repositório: .../src/risk_models/config.py -> sobe três níveis.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Pasta padrão dos YAMLs de configuração.
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

#: Dias úteis por ano — convenção usada na anualização de volatilidade.
TRADING_DAYS_PER_YEAR: int = 252


class DataSource(str, Enum):
    """Fonte de preços/volumes usada pela camada de dados.

    * ``YFINANCE``  — baixa dados reais via biblioteca ``yfinance``;
    * ``SYNTHETIC`` — gera dados simulados reprodutíveis (GBM), para
      testes/offline;
    * ``AUTO``      — tenta ``yfinance`` e, em caso de falha de rede,
      cai para o gerador sintético com aviso em log.
    """

    YFINANCE = "yfinance"
    SYNTHETIC = "synthetic"
    AUTO = "auto"


class VarMethod(str, Enum):
    """Metodologias de VaR suportadas em :mod:`risk_models.var`."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"


# ---------------------------------------------------------------------------
# Blocos de configuração (um dataclass por arquivo YAML)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathsConfig:
    """Diretórios do projeto (``configs/paths.yaml``).

    Cada campo pode ser sobrescrito pela variável de ambiente indicada,
    o que reproduz o comportamento descrito no README original
    (ex.: ``export RISK_CURATED_DIR=/dados/curados``).
    """

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    curated_dir: Path = PROJECT_ROOT / "data" / "curated_risk_pipeline"
    tables_dir: Path = PROJECT_ROOT / "outputs" / "tables"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "report"
    logs_dir: Path = PROJECT_ROOT / "logs"

    #: Mapeamento campo -> variável de ambiente que o sobrescreve.
    ENV_OVERRIDES = {
        "raw_dir": "RISK_RAW_DIR",
        "curated_dir": "RISK_CURATED_DIR",
        "tables_dir": "RISK_TABLES_DIR",
        "reports_dir": "RISK_REPORTS_DIR",
        "logs_dir": "RISK_LOGS_DIR",
    }

    def ensure_directories(self) -> None:
        """Cria todos os diretórios configurados (``mkdir -p``).

        Chamado uma vez no início do pipeline para que nenhuma tarefa
        falhe por pasta inexistente (robustez — ETAPA 3).
        """
        for directory in (self.raw_dir, self.curated_dir, self.tables_dir, self.reports_dir, self.logs_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LimitsConfig:
    """Limites de risco monitorados (``configs/limits.yaml``).

    O pipeline não bloqueia execução ao violar um limite — ele **alerta**
    (log ``WARNING`` e coluna ``breach`` no relatório), pois a decisão de
    reduzir posição é humana. Manter limites em arquivo, e não no código,
    permite que a área de risco os altere sem novo deploy.
    """

    var_99_max: float = 0.05  # VaR 99% máximo tolerado (fração do PL, ex.: 5%)
    cvar_99_max: float = 0.07
    dtl_max_days: float = 5.0  # dias para liquidar posição sob participação-alvo
    garch_persistence_max: float = 0.999  # alpha+beta < 1 exigido p/ estacionariedade
    min_history_days: int = 252  # histórico mínimo p/ estimar modelos com dignidade

    def validate(self) -> None:
        """Valida coerência interna dos limites.

        Raises
        ------
        ConfigurationError
            Se algum limite for não positivo ou logicamente impossível
            (ex.: CVaR máximo menor que VaR máximo).
        """
        if self.var_99_max <= 0 or self.cvar_99_max <= 0:
            raise ConfigurationError("Limites de VaR/CVaR devem ser positivos.")
        if self.cvar_99_max < self.var_99_max:
            # CVaR >= VaR por construção; um limite invertido indica erro de digitação.
            raise ConfigurationError(
                "cvar_99_max não pode ser menor que var_99_max (CVaR >= VaR por definição)."
            )
        if self.dtl_max_days <= 0:
            raise ConfigurationError("dtl_max_days deve ser positivo.")
        if not 0 < self.garch_persistence_max <= 1:
            raise ConfigurationError("garch_persistence_max deve estar em (0, 1].")
        if self.min_history_days < 30:
            raise ConfigurationError("min_history_days menor que 30 é estatisticamente frágil.")


@dataclass(frozen=True)
class PipelineParams:
    """Parâmetros do pipeline (``configs/pipeline.yaml``)."""

    tickers: tuple[str, ...] = ("PETR4.SA", "VALE3.SA", "ITUB4.SA", "BOVA11.SA")
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    start_date: str = "2020-01-02"
    end_date: str | None = None  # None => até hoje
    var_alphas: tuple[float, ...] = (0.95, 0.99)
    adv_window: int = 21  # ~1 mês útil, convenção de mercado p/ ADV
    corr_window: int = 63  # ~1 trimestre útil p/ correlação móvel
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    participation_rate: float = 0.20  # % do ADV consumível por dia sem impacto relevante
    spread_bps: float = 10.0  # spread bid-ask médio assumido (pontos-base)
    portfolio_value: float = 10_000_000.0  # PL de referência p/ métricas em moeda
    pca_components: int = 3
    data_source: DataSource = DataSource.AUTO
    synthetic_seed: int = 42  # reprodutibilidade do fallback sintético

    def validate(self) -> None:
        """Valida o bloco de parâmetros do pipeline.

        Raises
        ------
        ConfigurationError
            Para qualquer combinação inválida (detalhada na mensagem).
        """
        if not self.tickers:
            raise ConfigurationError("Lista de tickers vazia.")
        if len(self.weights) != len(self.tickers):
            raise ConfigurationError(
                f"{len(self.weights)} pesos para {len(self.tickers)} tickers — devem casar 1:1."
            )
        if any(w < 0 for w in self.weights):
            raise ConfigurationError("Pesos negativos não são suportados (portfólio long-only).")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-6):
            raise ConfigurationError(f"Pesos somam {sum(self.weights):.6f}; esperado 1.0.")
        for alpha in self.var_alphas:
            if not 0.5 < alpha < 1.0:
                raise ConfigurationError(f"Nível de confiança inválido: {alpha} (use (0.5, 1)).")
        if self.adv_window <= 0 or self.corr_window <= 1:
            raise ConfigurationError("Janelas adv_window/corr_window devem ser positivas (>1 p/ corr).")
        if not 0 <= self.winsor_lower < self.winsor_upper <= 1:
            raise ConfigurationError("Percentis de winsorização inválidos (0 <= lower < upper <= 1).")
        if not 0 < self.participation_rate <= 1:
            raise ConfigurationError("participation_rate deve estar em (0, 1].")
        if self.spread_bps < 0:
            raise ConfigurationError("spread_bps não pode ser negativo.")
        if self.portfolio_value <= 0:
            raise ConfigurationError("portfolio_value deve ser positivo.")
        if self.pca_components < 1:
            raise ConfigurationError("pca_components deve ser >= 1.")


@dataclass(frozen=True)
class RiskConfig:
    """Agregado imutável com toda a configuração carregada e validada."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    params: PipelineParams = field(default_factory=PipelineParams)


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lê um YAML retornando ``{}`` se o arquivo não existir.

    Arquivo ausente não é erro: os padrões dos dataclasses assumem.
    YAML *malformado*, por outro lado, é erro de configuração real e
    vira :class:`ConfigurationError` com o caminho do arquivo na mensagem.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # noqa: PERF203 — clareza > micro-otimização
        raise ConfigurationError(f"YAML inválido em {path}: {exc}") from exc
    if not isinstance(content, Mapping):
        raise ConfigurationError(f"Esperado mapeamento chave:valor em {path}.")
    return dict(content)


def _load_paths(config_dir: Path) -> PathsConfig:
    """Monta :class:`PathsConfig` aplicando YAML e depois ambiente."""
    raw = _read_yaml(config_dir / "paths.yaml")
    values: dict[str, Path] = {}
    for field_name, env_name in PathsConfig.ENV_OVERRIDES.items():
        if env_name in os.environ:  # 1º: variável de ambiente
            values[field_name] = Path(os.environ[env_name])
        elif field_name in raw:  # 2º: YAML
            values[field_name] = Path(str(raw[field_name]))
        # 3º: padrão do dataclass (não adiciona nada)
    return PathsConfig(**values)


def _load_section(config_dir: Path, filename: str, cls: type) -> Any:
    """Instancia ``cls`` a partir de ``configs/<filename>``.

    Chaves desconhecidas no YAML geram erro imediato — melhor falhar na
    carga do que ignorar silenciosamente um limite digitado errado.
    """
    raw = _read_yaml(config_dir / filename)
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ConfigurationError(
            f"Chave(s) desconhecida(s) em {filename}: {sorted(unknown)}. Válidas: {sorted(known)}."
        )
    # Conversões de tipo pontuais (YAML entrega listas/strings genéricas).
    if "tickers" in raw:
        raw["tickers"] = tuple(str(t) for t in raw["tickers"])
    if "weights" in raw:
        raw["weights"] = tuple(float(w) for w in raw["weights"])
    if "var_alphas" in raw:
        raw["var_alphas"] = tuple(float(a) for a in raw["var_alphas"])
    if "data_source" in raw:
        try:
            raw["data_source"] = DataSource(str(raw["data_source"]).lower())
        except ValueError as exc:
            valid = [s.value for s in DataSource]
            raise ConfigurationError(f"data_source inválido: {raw['data_source']!r}. Use {valid}.") from exc
    return cls(**raw)


def load_config(config_dir: Path | str | None = None) -> RiskConfig:
    """Carrega, valida e retorna a configuração completa do projeto.

    Parameters
    ----------
    config_dir:
        Pasta contendo os YAMLs. Padrão: ``<raiz>/configs``.

    Returns
    -------
    RiskConfig
        Configuração imutável pronta para uso pelo pipeline.

    Raises
    ------
    ConfigurationError
        Se qualquer arquivo estiver malformado ou algum valor for inválido.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg.params.adv_window
    21
    """
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR

    paths = _load_paths(directory)
    limits: LimitsConfig = _load_section(directory, "limits.yaml", LimitsConfig)
    params: PipelineParams = _load_section(directory, "pipeline.yaml", PipelineParams)

    limits.validate()
    params.validate()

    return RiskConfig(paths=paths, limits=limits, params=params)
