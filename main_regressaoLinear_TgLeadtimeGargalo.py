# --- 1. Importação de Bibliotecas ---
# pandas: Essencial para manipulação e análise de dados tabulares (DataFrames).
# numpy: Fornece suporte para arrays e operações matemáticas de alto nível.
# sklearn: A principal biblioteca de Machine Learning em Python, contendo modelos, ferramentas de pré-processamento e métricas de avaliação.
# matplotlib.pyplot: Usada para criar gráficos e visualizações de dados.
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings

# Ignora o FutureWarning do pandas que é esperado em algumas operações.
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 2. Carregamento dos Dados ---
# Define o caminho do arquivo de dados fornecido pelo usuário.
DATA_FILE = "//dados_desafio.xlsx"

try:
    # Tenta ler o arquivo Excel. Se o arquivo não for encontrado, exibe uma mensagem de erro e encerra o programa.
    df = pd.read_excel(DATA_FILE)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_FILE}' não encontrado. Verifique o caminho.")
    exit()

# --- 3. Pré-processamento e Engenharia de Atributos ---
# Converte as colunas de data para o formato datetime
df['data_hora_pedido'] = pd.to_datetime(df['data_hora_pedido'])
df['pedido_entregue'] = pd.to_datetime(df['pedido_entregue'])

# Cria a variável target: o lead time do pedido em dias
# Insight: O 'lead time' é a métrica de tempo que o cliente realmente se importa, do pedido à entrega.
df['lead_time_do_pedido'] = (df['pedido_entregue'] - df['data_hora_pedido']).dt.days

# Remova pedidos que não foram entregues ou que possuem lead time negativo
# Insight: Este é um tratamento de dado crucial. Dados de pedidos incompletos ou com erros (lead time negativo) não podem ser usados para treinar o modelo.
df = df.loc[df['status_pedido'] == 'delivered'].copy()
df = df.loc[df['lead_time_do_pedido'] >= 0].copy()

# Crie features de tempo a partir da data do pedido original
# Insight: A data e hora do pedido podem influenciar o lead time. Ex: pedidos feitos em feriados ou fins de semana podem ter um lead time maior.
df['mes'] = df['data_hora_pedido'].dt.month
df['dia_do_ano'] = df['data_hora_pedido'].dt.dayofyear
df['dia_da_semana'] = df['data_hora_pedido'].dt.day_name()
df['hora_do_dia'] = df['data_hora_pedido'].dt.hour

# Engenharia de Atributos Avançada baseada no lead time
# Insight: A ordenação por 'id_item' e 'data' é crucial para calcular as features de lag (histórico) corretamente.
df = df.sort_values(by=['id_item', 'data_hora_pedido'])
# 'shift(1)' pega o lead time do dia anterior.
df['lead_time_dia_anterior'] = df.groupby('id_item')['lead_time_do_pedido'].shift(1)
# 'rolling()' calcula a média e o desvio padrão do lead time nos últimos 7 dias.
df['media_lead_time_7d'] = df.groupby('id_item')['lead_time_do_pedido'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
df['std_lead_time_7d'] = df.groupby('id_item')['lead_time_do_pedido'].transform(lambda x: x.rolling(window=7, min_periods=1).std())

# --- 4. Definição de Features e Target ---
# Algumas features do modelo anterior não se aplicam e foram removidas
features = [
    'preco', 'valor_frete', 'peso_produto', 'comprimento_produto',
    'altura_produto', 'largura_produto', 'categoria_produto',
    'tipo_pagamento', 'score_avaliacao', 'mes', 'dia_do_ano', 'dia_da_semana',
    'hora_do_dia', 'lead_time_dia_anterior', 'media_lead_time_7d', 'std_lead_time_7d'
]
target = 'lead_time_do_pedido'

# Garante que as colunas selecionadas existem no DataFrame e trata valores nulos
X = df[features]
y = df[target]

# Separando os dados numéricos e categóricos
numeric_features = [
    'preco', 'valor_frete', 'peso_produto', 'comprimento_produto',
    'altura_produto', 'largura_produto', 'score_avaliacao', 'mes', 'dia_do_ano',
    'hora_do_dia', 'lead_time_dia_anterior', 'media_lead_time_7d', 'std_lead_time_7d'
]
categorical_features = ['categoria_produto', 'tipo_pagamento', 'dia_da_semana']

# --- 5. Tratamento de Dados Faltantes ---
# Insight: Este é um tratamento de dado essencial. Preencher com 'desconhecido'
# evita que o OneHotEncoder falhe.
X.loc[:, numeric_features] = X[numeric_features].fillna(0)
X.loc[:, categorical_features] = X[categorical_features].fillna('desconhecido')

# --- 6. Criação das Pipelines do Modelo ---
# O ColumnTransformer aplica transformações diferentes a diferentes colunas.
preprocessor = ColumnTransformer(
    transformers=[
        # 'passthrough' não altera as colunas numéricas
        ('num', 'passthrough', numeric_features),
        # 'OneHotEncoder' converte as colunas categóricas em representações numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# O Pipeline encadeia o pré-processamento e o modelo de regressão em um único objeto.
# Insight: Usar um pipeline garante que o pré-processamento seja aplicado de forma consistente.
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# --- 7. Validação Cruzada K-Fold e Métricas de Avaliação ---
# A validação cruzada K-Fold divide os dados em 5 partes para treinar e testar o modelo.
# Insight: Isso fornece uma avaliação mais robusta da performance do modelo, evitando o 'overfitting'.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def mape_scorer_func(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    valid_indices = y_true != 0
    mape = np.mean(np.abs((y_true[valid_indices] - y_pred[valid_indices]) / y_true[valid_indices])) * 100
    return mape if np.isfinite(mape) else 1000

def correlation_scorer_func(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0
    return np.corrcoef(y_true, y_pred)[0, 1]

mape_scorer = make_scorer(mape_scorer_func, greater_is_better=False)
correlation_scorer = make_scorer(correlation_scorer_func, greater_is_better=True)

scores_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
scores_rmse = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
scores_mae = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
scores_mape = cross_val_score(model, X, y, cv=kf, scoring=mape_scorer)
scores_correlation = cross_val_score(model, X, y, cv=kf, scoring=correlation_scorer)

# --- 8. Apresentação e Geração de Arquivos ---
# Salva os resultados em uma string
results_text = f"--- Resultados da Validação Cruzada K-Fold ---\n"
results_text += f"R² (média): {np.mean(scores_r2):.4f}\n"
results_text += f"RMSE (média): {-np.mean(scores_rmse):.4f}\n"
results_text += f"MAE (média): {-np.mean(scores_mae):.4f}\n"
results_text += f"MAPE (média): {np.mean(scores_mape):.2f}%\n"
results_text += f"Correlação (média): {np.mean(scores_correlation):.4f}\n"

# Salva a string no arquivo de texto 'resultado_TargetLeadTime.txt'.
# Insight: O output em arquivo é essencial para documentação e compartilhamento.
with open('resultado_TargetLeadTime.txt', 'w') as f:
    f.write(results_text)
print(results_text)

# Treina o modelo final para visualização
print("\nTreinando o modelo final para visualização...")
model.fit(X, y)

# Faz as previsões para todo o dataset
y_pred = model.predict(X)

# Crie um gráfico de dispersão comparando valores reais e previstos
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5, s=20)
plt.title('Valores Reais vs. Valores Previstos (Regressão Linear)', fontsize=16)
plt.xlabel('Lead Time Real (dias)', fontsize=12)
plt.ylabel('Lead Time Previsto (dias)', fontsize=12)

# Adiciona a linha diagonal vermelha, que representa a previsão perfeita.
max_val = max(y.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], '--r', linewidth=2)

plt.grid(True)
# Salva o gráfico em um arquivo PNG
plt.savefig('previsoes_leadtime.png')
print("Gráfico 'previsoes_leadtime.png' gerado com sucesso.")

# Exibe o gráfico na tela
plt.show()
