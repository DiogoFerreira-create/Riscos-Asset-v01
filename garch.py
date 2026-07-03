"""Entrada/saída de dados do pipeline (Parquet).

Decisão de arquitetura
----------------------
As tarefas do pipeline **não** trocam DataFrames por XCom do Airflow:
cada tarefa lê/escreve arquivos Parquet em ``data/curated_risk_pipeline``.
Vantagens: (i) tarefas idempotentes e re-executáveis isoladamente;
(ii) XCom não foi projetado para payloads grandes; (iii) os artefatos
intermediários ficam auditáveis em disco — requisito natural de uma área
de risco. Parquet (via ``pyarrow``) preserva dtypes e índices de datas,
ao contrário de CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exceptions import DataValidationError
from .logging_setup import get_logger

log = get_logger(__name__)


def save_parquet(frame: pd.DataFrame, path: Path | str) -> Path:
    """Grava ``frame`` em Parquet, criando diretórios se necessário.

    Parameters
    ----------
    frame:
        DataFrame a persistir (índice é preservado).
    path:
        Caminho de destino ``.parquet``.

    Returns
    -------
    pathlib.Path
        O caminho gravado (conveniente para encadear em logs).

    Raises
    ------
    DataValidationError
        Se ``frame`` estiver vazio — gravar artefato vazio esconderia a
        falha da etapa anterior; preferimos falhar alto e cedo.
    """
    if frame.empty:
        raise DataValidationError(f"Recusando gravar DataFrame vazio em {path}.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, engine="pyarrow")
    log.info("Gravado %s (%d linhas x %d colunas).", destination, *frame.shape)
    return destination


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Lê um Parquet gerado por :func:`save_parquet`.

    Raises
    ------
    DataValidationError
        Se o arquivo não existir — a mensagem orienta qual tarefa
        anterior do pipeline deveria tê-lo produzido.
    """
    source = Path(path)
    if not source.exists():
        raise DataValidationError(
            f"Arquivo esperado não encontrado: {source}. "
            "Execute a tarefa anterior do pipeline que o produz."
        )
    frame = pd.read_parquet(source, engine="pyarrow")
    log.info("Lido %s (%d linhas x %d colunas).", source, *frame.shape)
    return frame
