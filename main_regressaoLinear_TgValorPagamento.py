# --- 1. Importação de Bibliotecas ---
# pandas: Essencial para manipulação e análise de dados tabulares (DataFrames).
# numpy: Fornece suporte para arrays e operações matemáticas de alto nível.
# sklearn: A principal biblioteca de Machine Learning em Python, contendo modelos, ferramentas de pré-processamento e métricas de avaliação.
# matplotlib.pyplot: Usada para criar gráficos e visualizações de dados.
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 2. Carregamento dos Dados ---
# Define o caminho do arquivo de dados fornecido pelo usuário.
DATA_FILE = "//dados_desafio.xlsx"

# Tenta ler o arquivo Excel. Se o arquivo não for encontrado, exibe uma mensagem de erro e encerra o programa.
try:
    df = pd.read_excel(DATA_FILE)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_FILE}' não encontrado. Verifique o caminho.")
    exit()

# --- 3. Pré-processamento e Engenharia de Atributos ---
# O pré-processamento é a etapa de limpar e preparar os dados brutos.
# A engenharia de atributos cria novas informações a partir das colunas existentes.

# Converte a coluna de data e hora para o formato datetime, que permite extrair informações de tempo.
df['data_hora_pedido'] = pd.to_datetime(df['data_hora_pedido'])

# Cria novas colunas (features) a partir da coluna de data, como mês, dia do ano, dia da semana e hora do dia.
# Isso ajuda o modelo a encontrar padrões sazonais (ex: pagamentos maiores em certos meses ou horários).
df['mes'] = df['data_hora_pedido'].dt.month
df['dia_do_ano'] = df['data_hora_pedido'].dt.dayofyear
df['dia_da_semana'] = df['data_hora_pedido'].dt.day_name()
df['hora_do_dia'] = df['data_hora_pedido'].dt.hour

# Ordena os dados por item e data. Isso é crucial para o cálculo das features de "lag" (valores anteriores).
df = df.sort_values(by=['id_item', 'data_hora_pedido'])

# Cria features de "lag" para capturar o histórico de vendas.
# 'pagamento_dia_anterior': O valor do pagamento do pedido anterior do mesmo item.
# 'media_pagamento_7d': A média do valor de pagamento nos últimos 7 dias.
# 'std_pagamento_7d': O desvio padrão do valor de pagamento nos últimos 7 dias.
# Essas features dão ao modelo um "senso" de tendência e variabilidade, melhorando a previsão.
df['pagamento_dia_anterior'] = df.groupby('id_item')['valor_pagamento'].shift(1)
df['media_pagamento_7d'] = df.groupby('id_item')['valor_pagamento'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
df['std_pagamento_7d'] = df.groupby('id_item')['valor_pagamento'].transform(lambda x: x.rolling(window=7, min_periods=1).std())

# --- 4. Definição de Features e Variável-Alvo (Target) ---

# 'features' são as variáveis de entrada que o modelo usará para fazer a previsão.
features = [
    'prestacoes', 'preco', 'valor_frete', 'peso_produto', 'comprimento_produto',
    'altura_produto', 'largura_produto', 'score_avaliacao', 'categoria_produto',
    'tipo_pagamento', 'status_pedido', 'mes', 'dia_do_ano', 'dia_da_semana',
    'hora_do_dia', 'pagamento_dia_anterior', 'media_pagamento_7d', 'std_pagamento_7d'
]
# 'target' é a variável que queremos prever, que é o valor de pagamento.
target = 'valor_pagamento'

# Divide os dados em features (X) e target (y).
X = df[features]
y = df[target]

# --- 5. Tratamento de Dados Faltantes ---

# Separa as features numéricas e categóricas para tratamento.
numeric_features = [
    'prestacoes', 'preco', 'valor_frete', 'peso_produto', 'comprimento_produto',
    'altura_produto', 'largura_produto', 'score_avaliacao', 'mes', 'dia_do_ano',
    'hora_do_dia', 'pagamento_dia_anterior', 'media_pagamento_7d', 'std_pagamento_7d'
]
categorical_features = ['categoria_produto', 'tipo_pagamento', 'status_pedido', 'dia_da_semana']

# Preenche os valores ausentes (NaN) nas colunas numéricas com 0.
X.loc[:, numeric_features] = X[numeric_features].fillna(0)
# Preenche os valores ausentes nas colunas categóricas com a string 'desconhecido'.
# Isso é crucial para que o OneHotEncoder funcione corretamente, sem misturar tipos.
X.loc[:, categorical_features] = X[categorical_features].fillna('desconhecido')

# --- 6. Criação de Pipelines de Pré-processamento e Modelo ---

# 'ColumnTransformer' aplica transformações diferentes a diferentes tipos de colunas.
preprocessor = ColumnTransformer(
    transformers=[
        # 'passthrough' para manter as colunas numéricas como estão, sem transformá-las.
        ('num', 'passthrough', numeric_features),
        # 'OneHotEncoder' converte as colunas categóricas em um formato numérico.
        # 'handle_unknown='ignore'' evita erros se novas categorias aparecerem.
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 'Pipeline' encadeia as etapas de pré-processamento e o modelo.
# Isso garante que a mesma transformação de dados seja aplicada de forma consistente ao treino e teste.
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# --- 7. Validação Cruzada K-Fold e Métricas de Avaliação ---

# 'KFold' divide o dataset em 5 "dobras" (folds) para testar o modelo de forma mais robusta.
# Cada dobra será usada como um conjunto de teste, enquanto as outras 4 são usadas para treino.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Funções personalizadas para métricas específicas.
# MAPE: calcula o erro percentual, ignorando a divisão por zero.
def mape_scorer_func(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    valid_indices = y_true != 0
    mape = np.mean(np.abs((y_true[valid_indices] - y_pred[valid_indices]) / y_true[valid_indices])) * 100
    return mape if np.isfinite(mape) else 1000

# Correlação: mede a relação linear entre os valores reais e previstos.
def correlation_scorer_func(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0
    return np.corrcoef(y_true, y_pred)[0, 1]

# 'make_scorer' converte as funções personalizadas para um formato compatível com o cross_val_score.
mape_scorer = make_scorer(mape_scorer_func, greater_is_better=False)
correlation_scorer = make_scorer(correlation_scorer_func, greater_is_better=True)

# 'cross_val_score' avalia o modelo em cada dobra e retorna a média das métricas.
scores_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
scores_rmse = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
scores_mae = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
scores_mape = cross_val_score(model, X, y, cv=kf, scoring=mape_scorer)
scores_correlation = cross_val_score(model, X, y, cv=kf, scoring=correlation_scorer)

# --- 8. Apresentação e Geração de Arquivos de Resultados ---

# Cria uma string com os resultados da validação cruzada para imprimir e salvar.
results_text = f"--- Resultados da Validação Cruzada K-Fold ---\n"
results_text += f"R² (média): {np.mean(scores_r2):.4f}\n"
results_text += f"RMSE (média): {-np.mean(scores_rmse):.4f}\n"
results_text += f"MAE (média): {-np.mean(scores_mae):.4f}\n"
results_text += f"MAPE (média): {np.mean(scores_mape):.2f}%\n"
results_text += f"Correlação (média): {np.mean(scores_correlation):.4f}\n"

# Salva a string no arquivo de texto 'resultados.txt'.
with open('resultado_TargetValorPagamento.txt', 'w') as f:
    f.write(results_text)

# Imprime os resultados no terminal.
print(results_text)

# Treina o modelo final usando todos os dados. Isso é feito apenas para a visualização,
# pois o cross_val_score não retorna um modelo treinado.
print("\nTreinando o modelo final para visualização...")
model.fit(X, y)

# Faz as previsões para todo o dataset.
y_pred = model.predict(X)

# Crie o gráfico de dispersão que compara os valores reais com os previstos.
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5, s=20)
plt.title('Valores Reais vs. Valores Previstos (Regressão Linear)', fontsize=16)
plt.xlabel('Valores Reais do Pagamento', fontsize=12)
plt.ylabel('Valores Previstos pelo Modelo', fontsize=12)

# Adiciona a linha diagonal vermelha, que representa a previsão perfeita.
# Quanto mais próximos os pontos estiverem desta linha, melhor é o modelo.
max_val = max(y.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], '--r', linewidth=2)

plt.grid(True)

# Salva o gráfico em um arquivo PNG.
plt.savefig('previsoes.png')
print("Gráfico 'previsoes.png' gerado com sucesso.")

# Exibe o gráfico na tela.
plt.show()
