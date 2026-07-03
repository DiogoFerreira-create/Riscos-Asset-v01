"""Hierarquia de exceções do projeto Riscos-Asset.

Todas as exceções levantadas pelo pacote ``risk_models`` derivam de
:class:`RiskModelError`. Isso permite que o código cliente (por exemplo, o
pipeline do Airflow) capture qualquer falha do domínio com um único
``except RiskModelError`` sem engolir bugs de programação (``TypeError``,
``KeyError`` etc.), que continuam propagando normalmente.
"""

from __future__ import annotations


class RiskModelError(Exception):
    """Erro-base para todas as falhas de domínio do pacote ``risk_models``."""


class ConfigurationError(RiskModelError):
    """Configuração ausente, malformada ou com valores inválidos.

    Levantada ao carregar/validar os arquivos YAML de ``configs/`` ou
    variáveis de ambiente que os sobrescrevem.
    """


class DataValidationError(RiskModelError):
    """Dados de entrada não atendem aos pré-requisitos do modelo.

    Exemplos: série vazia, preços não positivos para retorno logarítmico,
    janela maior que o histórico disponível, pesos que não somam 1.
    """


class DataSourceError(RiskModelError):
    """Falha ao obter dados de mercado da fonte configurada.

    Exemplo: ``yfinance`` indisponível/sem rede e fallback sintético
    desabilitado na configuração.
    """


class ModelFitError(RiskModelError):
    """Falha na estimação de um modelo estatístico (ex.: GARCH não converge)."""
