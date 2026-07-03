# ---------------------------------------------------------------------------
# Parâmetros do pipeline: universo, pesos, janelas e fonte de dados.
# ---------------------------------------------------------------------------
tickers: [PETR4.SA, VALE3.SA, ITUB4.SA, BOVA11.SA]
weights: [0.25, 0.25, 0.25, 0.25]     # devem somar 1.0
start_date: "2020-01-02"
# end_date: "2025-12-31"              # omitido => até hoje
var_alphas: [0.95, 0.99]
adv_window: 21                        # ~1 mês útil
corr_window: 63                       # ~1 trimestre útil
winsor_lower: 0.01
winsor_upper: 0.99
participation_rate: 0.20              # % do ADV consumível por dia
spread_bps: 10.0                      # spread bid-ask médio assumido
portfolio_value: 10000000.0           # PL de referência (R$ 10 mi)
pca_components: 3
data_source: auto                     # yfinance | synthetic | auto
synthetic_seed: 42
