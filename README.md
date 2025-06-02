# Riscos-Asset-v01

# Projeto Riscos-Asset-v01

## Descrição Curta
O `Riscos-Asset-v01` é uma plataforma em Python para modelagem e monitoramento de riscos de mercado e liquidez em portfólios de ativos. O sistema automatiza o cálculo de métricas essenciais, orquestrado por um pipeline de dados diário, com foco em configuração flexível, testes e documentação para apoiar o controle de riscos e a tomada de decisão.

## Visão Geral
Este projeto implementa um conjunto de modelos quantitativos para análise de risco financeiro. Ele permite a extração de dados de mercado, o cálculo de diversas métricas de risco, a modelagem de volatilidade, a análise de liquidez e fatores, além do backtesting dos modelos de VaR. A orquestração das tarefas é projetada para ser gerenciada pelo Apache Airflow, garantindo execuções diárias e consistentes.

## Principais Funcionalidades e Módulos Implementados (`risk_models`)
* **Retornos (`returns`):**
    * Download de preços históricos de ativos (via `yfinance`).
    * Cálculo de retornos logarítmicos.
    * Tratamento de outliers (Winsorização).
* **Value-at-Risk (`var`):**
    * VaR Histórico.
    * VaR Paramétrico (Delta-Normal).
    * Conditional VaR (CVaR) / Expected Shortfall.
* **Volatilidade (`garch`):**
    * Ajuste de modelos GARCH (padrão GARCH(1,1)).
    * Previsão de volatilidade um passo à frente.
* **Liquidez (`liquidity`):**
    * Cálculo de Volume Médio Diário Negociado (ADV).
    * Cálculo de Dias para Liquidar (DTL).
    * Estimativa de Slippage de Bid-Ask.
* **Fatores de Risco (`factors`):**
    * Cálculo de matrizes de correlação (móvel ou amostra completa).
    * Análise de Componentes Principais (PCA) para extração de fatores.
* **Backtesting de VaR (`backtest`):**
    * Teste de Proporção de Falhas de Kupiec (POF / LR_uc).
    * Teste de Cobertura Condicional de Christoffersen (LR_cc).

## Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Bibliotecas Principais:**
    * Pandas: Manipulação e análise de dados.
    * NumPy: Operações numéricas.
    * SciPy: Funções estatísticas (distribuições, testes Qui-quadrado).
    * yfinance: Download de dados de mercado.
    * arch: Modelagem GARCH.
    * scikit-learn: Análise de Componentes Principais (PCA).
    * PyArrow: Leitura e escrita de arquivos Parquet.
    * Pendulum: Manipulação de datas e fusos horários (especialmente para Airflow).
* **Orquestração de Pipeline:** Apache Airflow (o script da DAG está em `dags/risk_pipeline.py`).
* **Testes:** Pytest.
* **Configuração:** Arquivos YAML (`config/paths.yaml`, `config/limits.yaml`).

## Estrutura do Projeto (Principais Pastas)
```
Riscos-Asset-v01/
├── config/               # Arquivos de configuração (YAML) para paths e limites
├── dags/                 # Scripts de DAGs para Apache Airflow
│   └── risk_pipeline.py  # Pipeline principal de cálculo de riscos
├── data/                 # Dados gerados pelo pipeline (ex: Parquet)
│   └── curated_risk_pipeline/
├── risk_models/          # Pacote Python com os módulos de cálculo de risco
│   ├── __init__.py       # Torna risk_models um pacote e define a API pública
│   ├── returns.py
│   ├── var.py
│   ├── garch.py
│   ├── liquidity.py
│   ├── factors.py
│   └── backtest.py
├── tests/                # Testes unitários (PyTest)
│   ├── test_var.py
│   ├── test_liquidity.py
│   └── test_backtest.py
├── requirements.txt      # Dependências do projeto Python
├── MODEL_RISK.md         # Documentação técnica dos modelos
└── README.md             # Este arquivo
```

## Configuração e Execução

### 1. Pré-requisitos
* Python 3.8+ (ou a versão especificada no seu ambiente)
* `pip` (gerenciador de pacotes Python)

### 2. Instalação de Dependências
Clone o repositório (se aplicável) ou certifique-se de que todos os arquivos do projeto estejam no seu ambiente. Na pasta raiz do projeto (`Riscos-Asset-v01/`), instale as dependências:
```bash
pip install -r requirements.txt
```

### 3. Configuração
* Os caminhos para dados (`raw`, `curated`, `tables`, etc.) e os limites de risco podem ser configurados através de variáveis de ambiente. Consulte os arquivos `config/paths.yaml` e `config/limits.yaml` para ver os nomes das variáveis e os valores padrão.
* Por exemplo, para definir o diretório de dados curados, você pode exportar no seu terminal:
    ```bash
    export RISK_CURATED_DIR="/caminho/para/seus/dados/curados"
    ```
    Se as variáveis de ambiente não estiverem definidas, o pipeline usará padrões definidos no código (geralmente criando uma pasta `data/curated_risk_pipeline/` dentro da raiz do projeto para os arquivos Parquet gerados na execução local de teste).

### 4. Executando os Testes Unitários
Para rodar os testes unitários (requer `pytest` instalado):
```bash
pytest
```
Ou para um arquivo de teste específico (estando na raiz do projeto):
```bash
pytest tests/nome_do_arquivo_de_teste.py
```

### 5. Executando o Pipeline
* **Localmente (para teste das tarefas Python, sem um ambiente Airflow completo):**
    O script `dags/risk_pipeline.py` contém um bloco `if __name__ == "__main__":` (que é ativado quando o Airflow não é detectado) que permite executar uma sequência de tarefas para fins de teste e depuração local. Execute a partir da pasta raiz do projeto:
    ```bash
    python dags/risk_pipeline.py
    ```
    (Este comando assume que o ajuste de `sys.path` dentro de `dags/risk_pipeline.py` está funcionando para encontrar o pacote `risk_models`.)
* **Com Apache Airflow:**
    1.  Certifique-se de que o Airflow esteja instalado e configurado corretamente.
    2.  Coloque (ou crie um link simbólico para) o arquivo `dags/risk_pipeline.py` na pasta de DAGs (`dags_folder`) do seu ambiente Airflow.
    3.  O Airflow irá detectar o DAG automaticamente. Ele poderá então ser habilitado, gerenciado e executado através da interface web do Airflow, seguindo o agendamento definido (`08:00 America/Recife, Segunda a Sexta`).

## Documentação dos Modelos
Consulte o arquivo [MODEL_RISK.md](MODEL_RISK.md) para uma descrição técnica detalhada das fórmulas, hipóteses e limitações dos modelos implementados no pacote `risk_models`.

## Próximos Passos / Funcionalidades Futuras
* **Dashboards Interativos:** Em breve, o projeto contará com dashboards interativos (potencialmente utilizando ferramentas como Streamlit, Dash ou integrando com plataformas de Business Intelligence) para visualização dinâmica das métricas de risco e dos resultados dos modelos.
* **Testes de Estresse (Stress Testing):** Planejamos expandir as capacidades de análise de risco com a implementação de cenários de testes de estresse mais elaborados e configuráveis, permitindo uma avaliação mais profunda do impacto de eventos de mercado extremos no portfólio.
* Refinamento contínuo da estratégia de cálculo de VaR e Retorno de Portfólio dentro da tarefa `task_var_backtest`.
* Integração de uma fonte de dados de volume real para a tarefa `task_liquidity_metrics`, substituindo o placeholder atual.
