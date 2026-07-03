"""Dashboard interativo de risco (Plotly -> HTML autocontido).

O que este módulo produz
------------------------
Um único arquivo ``risk_dashboard.html`` com todos os gráficos do dia,
**interativo** (zoom, pan, hover com valores, liga/desliga séries pela
legenda) e **autocontido**: a biblioteca ``plotly.js`` é embutida no
próprio arquivo, então ele abre em qualquer navegador *sem internet e
sem servidor* — dá para anexar no e-mail do comitê de risco.

Por que HTML estático e não um app (Streamlit/Dash)?
----------------------------------------------------
O pipeline é um processo *batch* diário; seu artefato natural é um
arquivo, não um serviço que precisa ficar de pé. O HTML entra na mesma
trilha auditável dos Parquets (o dashboard de terça é o dashboard de
terça para sempre). Um app interativo com widgets continua sendo boa
evolução (Capítulo 10 da documentação) — e reaproveitaria as funções
``figure_*`` deste módulo sem alteração.

Robustez
--------
Cada seção é construída de forma independente: se um artefato estiver
ausente/corrompido, a seção correspondente vira um aviso visível no
HTML e o restante do dashboard é gerado normalmente (mesma filosofia de
isolamento de falhas do ``garch_report``).
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .exceptions import DataValidationError
from .logging_setup import get_logger
from .validation import require_alpha

log = get_logger(__name__)

# Paleta sóbria e consistente entre gráficos (ordem estável por ativo).
_PALETTE = ("#123047", "#1d6f8f", "#c05b2a", "#5a7d2a", "#7a4b8f",
            "#a3323b", "#2a8f7a", "#8f6f1d")

#: Template visual comum: fundo branco, fonte legível, margens enxutas.
_LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Helvetica, Arial, sans-serif", size=13),
    margin=dict(l=60, r=30, t=60, b=50),
    hovermode="x unified",
)


def _apply_base(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Aplica o layout-padrão do projeto a uma figura (título, altura)."""
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=17)),
                      height=height, **_LAYOUT_BASE)
    return fig


# ---------------------------------------------------------------------------
# Figuras individuais (funções puras: DataFrame entra, go.Figure sai)
# ---------------------------------------------------------------------------


def figure_normalized_prices(prices: pd.DataFrame) -> go.Figure:
    """Preços normalizados (base 100 na primeira data) por ativo.

    Base 100 coloca ativos de preços muito diferentes (R\\$ 12 e R\\$ 90)
    na mesma régua de *desempenho acumulado* — a pergunta que o leitor
    de um dashboard realmente faz.
    """
    if prices.empty:
        raise DataValidationError("Preços vazios: nada a plotar.")
    normalized = 100.0 * prices / prices.iloc[0]
    fig = go.Figure()
    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col], name=str(col),
            line=dict(width=2, color=_PALETTE[i % len(_PALETTE)]),
            hovertemplate="%{y:.1f}<extra>" + str(col) + "</extra>",
        ))
    fig.update_yaxes(title="Índice (base 100)")
    return _apply_base(fig, "Desempenho acumulado — preços normalizados")


def figure_portfolio_var(port_returns: pd.Series,
                         var_by_alpha: dict[float, float]) -> go.Figure:
    """Retornos diários do portfólio com linhas de -VaR e violações.

    As violações (retorno abaixo de -VaR) são destacadas em marcador
    próprio por nível de confiança — é a visualização direta do que os
    testes de Kupiec/Christoffersen medem numericamente.
    """
    if port_returns.empty:
        raise DataValidationError("Retornos do portfólio vazios.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_returns.index, y=port_returns, name="Retorno diário",
        mode="lines", line=dict(width=1, color="#8a9bab"),
        hovertemplate="%{y:.2%}<extra>retorno</extra>",
    ))
    dash_styles = {0.95: "dot", 0.99: "dash"}
    marker_colors = {0.95: "#c05b2a", 0.99: "#a3323b"}
    for alpha, var_value in sorted(var_by_alpha.items()):
        require_alpha(alpha)
        tag = f"{int(round(alpha * 100))}%"
        color = marker_colors.get(alpha, "#a3323b")
        fig.add_hline(y=-var_value, line_dash=dash_styles.get(alpha, "dash"),
                      line_color=color,
                      annotation_text=f"-VaR {tag} ({var_value:.2%})",
                      annotation_position="bottom right")
        breaches = port_returns[port_returns < -var_value]
        fig.add_trace(go.Scatter(
            x=breaches.index, y=breaches, mode="markers",
            name=f"Violações VaR {tag} ({len(breaches)})",
            marker=dict(size=7, color=color, symbol="x"),
            hovertemplate="%{y:.2%}<extra>violação " + tag + "</extra>",
        ))
    fig.update_yaxes(title="Retorno diário", tickformat=".1%")
    return _apply_base(fig, "Portfólio: retornos, VaR e violações", height=460)


def figure_var_by_asset(var_report: pd.DataFrame) -> go.Figure:
    """Barras agrupadas de VaR histórico/paramétrico e CVaR por ativo."""
    if var_report.empty:
        raise DataValidationError("Relatório de VaR vazio.")
    assets = [ix for ix in var_report.index if ix != "PORTFOLIO"] + ["PORTFOLIO"]
    metrics = [c for c in ("var_hist_99", "var_param_99", "cvar_99")
               if c in var_report.columns]
    labels = {"var_hist_99": "VaR 99% histórico",
              "var_param_99": "VaR 99% paramétrico", "cvar_99": "CVaR 99%"}
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=assets, y=[var_report.loc[a, metric] for a in assets],
            name=labels[metric], marker_color=_PALETTE[i % len(_PALETTE)],
            hovertemplate="%{y:.2%}<extra>" + labels[metric] + "</extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Perda potencial diária", tickformat=".1%")
    return _apply_base(fig, "VaR e CVaR (99%) por ativo")


def figure_garch(garch_report: pd.DataFrame) -> go.Figure:
    """Vol prevista (anualizada) nas barras + persistência nos marcadores.

    Dois eixos porque as grandezas têm escalas diferentes (vol em %,
    persistência em [0,1]) mas leem-se juntas: vol alta E persistente é
    o quadrante que merece atenção.
    """
    if garch_report.empty:
        raise DataValidationError("Relatório GARCH vazio.")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=garch_report.index, y=garch_report["forecast_vol_annual"],
        name="Vol prevista (anual)", marker_color=_PALETTE[1],
        hovertemplate="%{y:.1%}<extra>vol anual prevista</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=garch_report.index, y=garch_report["persistence"],
        name="Persistência (α+β)", mode="markers+text", yaxis="y2",
        marker=dict(size=12, color=_PALETTE[2], symbol="diamond"),
        text=[f"{v:.2f}" for v in garch_report["persistence"]],
        textposition="top center",
        hovertemplate="%{y:.3f}<extra>persistência</extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="Vol anualizada prevista", tickformat=".0%"),
        yaxis2=dict(title="Persistência (α+β)", overlaying="y", side="right",
                    range=[0, 1.05], showgrid=False),
    )
    return _apply_base(fig, "GARCH(1,1): volatilidade prevista e persistência")


def figure_correlation(corr: pd.DataFrame) -> go.Figure:
    """Heatmap da matriz de correlação com valores anotados."""
    if corr.empty:
        raise DataValidationError("Matriz de correlação vazia.")
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.astype(str), y=corr.index.astype(str),
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
        colorbar=dict(title="ρ"),
        hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
    ))
    return _apply_base(fig, "Correlação (janela móvel)", height=440)


def figure_pca(variance: pd.Series, loadings: pd.DataFrame) -> go.Figure:
    """Variância explicada (barras, com acumulada) + cargas (heatmap).

    Os dois painéis respondem em sequência: *quantos* fatores importam
    (esquerda) e *o que* cada fator é (direita).
    """
    from plotly.subplots import make_subplots

    if variance.empty or loadings.empty:
        raise DataValidationError("Resultados de PCA vazios.")
    fig = make_subplots(cols=2, rows=1, column_widths=[0.42, 0.58],
                        subplot_titles=("Variância explicada", "Cargas (loadings)"),
                        horizontal_spacing=0.14)
    cumulative = variance.cumsum()
    fig.add_trace(go.Bar(
        x=variance.index, y=variance.values, name="por componente",
        marker_color=_PALETTE[0],
        hovertemplate="%{y:.1%}<extra>variância</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values, name="acumulada",
        mode="lines+markers", line=dict(color=_PALETTE[2]),
        hovertemplate="%{y:.1%}<extra>acumulada</extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=loadings.values, x=loadings.columns.astype(str),
        y=loadings.index.astype(str), colorscale="RdBu", reversescale=True,
        zmid=0, colorbar=dict(title="carga", x=1.02),
        hovertemplate="%{y} em %{x}: %{z:.2f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    return _apply_base(fig, "Fatores de risco — PCA", height=430)


def figure_liquidity(liquidity_report: pd.DataFrame,
                     dtl_limit: float | None = None) -> go.Figure:
    """DTL por ativo (barras), com linha do limite configurado.

    Barras que cruzam a linha são exatamente as que aparecerão como
    ``breach`` no consolidado — coerência visual com o CSV.
    """
    if liquidity_report.empty:
        raise DataValidationError("Relatório de liquidez vazio.")
    dtl = liquidity_report["dtl_days"].replace([np.inf], np.nan)
    illiquid = liquidity_report.index[liquidity_report["dtl_days"].apply(np.isinf)]
    colors = [_PALETTE[5] if (dtl_limit is not None and v > dtl_limit)
              else _PALETTE[1] for v in dtl.fillna(0)]
    fig = go.Figure(go.Bar(
        x=dtl.index.astype(str), y=dtl.values, marker_color=colors,
        name="DTL (dias)",
        customdata=np.stack([liquidity_report["adv"].values,
                             liquidity_report["slippage_cost"].values], axis=-1),
        hovertemplate=("DTL: %{y:.0f} dias<br>ADV: R$ %{customdata[0]:,.0f}"
                       "<br>Slippage: R$ %{customdata[1]:,.0f}<extra></extra>"),
    ))
    if dtl_limit is not None:
        fig.add_hline(y=dtl_limit, line_dash="dash", line_color="#a3323b",
                      annotation_text=f"limite {dtl_limit:.0f} dias")
    title = "Liquidez: dias para liquidar (DTL)"
    if len(illiquid) > 0:
        title += f" — ilíquidos (DTL ∞): {', '.join(map(str, illiquid))}"
    fig.update_yaxes(title="Dias")
    return _apply_base(fig, title)


def figure_limits_table(summary: pd.DataFrame) -> go.Figure:
    """Tabela executiva de limites com violações destacadas em vermelho."""
    if summary.empty:
        raise DataValidationError("Resumo de limites vazio.")
    breach_colors = ["#f8d7da" if b else "#e8f5e0" for b in summary["breach"]]
    fig = go.Figure(go.Table(
        header=dict(values=["Métrica", "Valor", "Limite", "Violado?"],
                    fill_color="#123047", font=dict(color="white", size=13),
                    align="left"),
        cells=dict(
            values=[summary["metric"],
                    [f"{v:.4f}" for v in summary["value"]],
                    [f"{v:.4f}" for v in summary["limit"]],
                    ["SIM" if b else "não" for b in summary["breach"]]],
            fill_color=[["white"] * len(summary)] * 3 + [breach_colors],
            align="left", height=28),
    ))
    return _apply_base(fig, "Checagem de limites de risco", height=240)


# ---------------------------------------------------------------------------
# Montagem do HTML
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Riscos-Asset — Dashboard de Risco</title>
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; margin: 0; background: #f4f6f8; color: #1a2733; }}
  header {{ background: #123047; color: white; padding: 18px 32px; }}
  header h1 {{ margin: 0; font-size: 22px; }}
  header p {{ margin: 4px 0 0; opacity: .8; font-size: 13px; }}
  main {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
  section {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
            margin-bottom: 22px; padding: 10px 14px; }}
  .aviso {{ background: #fff3cd; border-left: 5px solid #c05b2a; padding: 12px 16px;
           border-radius: 6px; font-size: 14px; }}
  footer {{ text-align: center; font-size: 12px; color: #6b7b88; padding: 18px; }}
</style>
</head>
<body>
<header>
  <h1>Dashboard de Risco — Riscos-Asset-v01</h1>
  <p>Gerado em {generated_at} · fonte de dados: {source} · gráficos interativos (zoom, hover, clique na legenda)</p>
</header>
<main>
{sections}
</main>
<footer>Riscos-Asset-v01 · dashboard autocontido (plotly.js embutido) · abre offline em qualquer navegador</footer>
</body>
</html>
"""


def _section(fig: go.Figure, include_js: bool) -> str:
    """Converte figura em ``<section>`` HTML; a 1ª embute o plotly.js."""
    inner = fig.to_html(full_html=False,
                        include_plotlyjs="inline" if include_js else False,
                        config={"displaylogo": False, "locale": "pt-BR"})
    return f"<section>{inner}</section>"


def _warning_section(name: str, error: Exception) -> str:
    """Seção de aviso quando um gráfico não pôde ser construído."""
    return (f'<section><div class="aviso"><strong>{html.escape(name)}</strong>: '
            f"gráfico indisponível — {html.escape(str(error))}</div></section>")


def build_dashboard_html(
    figures: list[tuple[str, Callable[[], go.Figure]]],
    output_path: Path | str,
    source_label: str = "desconhecida",
) -> Path:
    """Monta o HTML final a partir de construtores de figura.

    Parameters
    ----------
    figures:
        Lista ordenada ``(nome_da_seção, função_sem_args -> go.Figure)``.
        Receber *construtores* (e não figuras prontas) permite isolar a
        falha de cada seção: a exceção de um gráfico vira aviso no HTML
        sem impedir os demais.
    output_path:
        Caminho do ``.html`` de saída (diretórios são criados).
    source_label:
        Texto exibido no cabeçalho (ex.: ``"yfinance"``/``"synthetic"``).

    Returns
    -------
    pathlib.Path
        Caminho gravado.

    Raises
    ------
    DataValidationError
        Se *nenhuma* seção puder ser construída (dashboard vazio não é
        entregável válido).
    """
    sections: list[str] = []
    built = 0
    js_pending = True
    for name, builder in figures:
        try:
            fig = builder()
        except Exception as exc:  # noqa: BLE001 — isolamento por seção é o objetivo
            log.error("Dashboard: seção '%s' falhou: %s", name, exc)
            sections.append(_warning_section(name, exc))
            continue
        sections.append(_section(fig, include_js=js_pending))
        js_pending = False
        built += 1
    if built == 0:
        raise DataValidationError("Nenhuma seção do dashboard pôde ser construída.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    page = _PAGE_TEMPLATE.format(
        generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
        source=html.escape(source_label),
        sections="\n".join(sections),
    )
    destination.write_text(page, encoding="utf-8")
    log.info("Dashboard interativo gravado em %s (%d seções, %.1f MB).",
             destination, built, destination.stat().st_size / 1e6)
    return destination
