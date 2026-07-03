"""Configuração centralizada de logging do projeto.

Motivação
---------
Cada módulo obtém seu logger via :func:`get_logger` em vez de chamar
``logging.basicConfig`` por conta própria. Isso garante formato único,
evita handlers duplicados quando módulos são importados múltiplas vezes
(problema clássico em DAGs do Airflow, que reimportam arquivos) e envia
uma cópia dos logs para arquivo rotativo em ``logs/`` — exigência de
auditoria comum em sistemas de risco.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Formato: timestamp | nível | módulo | mensagem — suficiente para auditoria
# sem poluir o console.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Flag de módulo para configurar handlers uma única vez por processo.
_CONFIGURED = False


def setup_logging(
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
    filename: str = "risk_pipeline.log",
) -> None:
    """Configura o logger raiz do projeto (idempotente).

    Parameters
    ----------
    log_dir:
        Diretório onde o arquivo de log rotativo será gravado. Se ``None``,
        apenas o handler de console é instalado (útil em testes).
    level:
        Nível mínimo de log (padrão ``logging.INFO``).
    filename:
        Nome do arquivo de log dentro de ``log_dir``.

    Notes
    -----
    Chamadas subsequentes são ignoradas (no-op) para impedir handlers
    duplicados — comportamento necessário porque o Airflow importa o
    arquivo da DAG a cada ciclo de parsing.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("risk_models")
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir)
        # Cria a pasta de logs se não existir: robustez pedida na ETAPA 3.
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / filename,
            maxBytes=5 * 1024 * 1024,  # 5 MB por arquivo
            backupCount=5,  # mantém histórico limitado, sem crescer sem fim
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger filho de ``risk_models`` com o nome do módulo.

    Parameters
    ----------
    name:
        Normalmente ``__name__`` do módulo chamador.

    Returns
    -------
    logging.Logger
        Logger hierárquico (ex.: ``risk_models.var``) que herda os
        handlers configurados em :func:`setup_logging`.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("VaR calculado")  # doctest: +SKIP
    """
    short = name.removeprefix("risk_models.").removeprefix("src.")
    return logging.getLogger(f"risk_models.{short}")
