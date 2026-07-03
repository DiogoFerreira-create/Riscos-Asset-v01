# ---------------------------------------------------------------------------
# Limites de risco monitorados pela tarefa de consolidação.
# Violações geram WARNING no log e coluna breach=True no risk_summary.csv.
# ---------------------------------------------------------------------------
var_99_max: 0.05            # VaR 99% diário máximo do portfólio (5% do PL)
cvar_99_max: 0.07           # CVaR 99% diário máximo (>= var_99_max)
dtl_max_days: 5.0           # pior DTL tolerado sob a participação-alvo
garch_persistence_max: 0.999  # alerta se alpha+beta chegar aqui
min_history_days: 252       # histórico mínimo desejado p/ estimação
