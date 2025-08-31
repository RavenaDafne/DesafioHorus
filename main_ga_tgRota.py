# --- 1. Importação de Bibliotecas ---
# pandas: Essencial para manipulação e análise de dados tabulares (DataFrames).
# numpy: Fornece suporte para arrays e operações matemáticas de alto nível.
# random: Usado para gerar aleatoriedade no algoritmo genético.
# math: Funções matemáticas para cálculos de distância.
# matplotlib.pyplot: Usada para criar gráficos e visualizações de dados.
# warnings: Para ignorar avisos que não afetam a execução.
# datetime: Para medir o tempo de execução do algoritmo.
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Ignora o FutureWarning do pandas que é esperado em algumas operações
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 2. Preparação dos Dados ---
# Define o caminho do arquivo de dados fornecido pelo usuário.
DATA_FILE = "//dados_desafio.xlsx"

try:
    # Tenta ler o arquivo Excel. Se não encontrar, exibe uma mensagem de erro e encerra o programa.
    # Insights de Negócio: A leitura direta de arquivos .xlsx é uma etapa fundamental para evitar erros de importação e iniciar a análise.
    df = pd.read_excel(DATA_FILE)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_FILE}' não encontrado. Verifique o caminho.")
    exit()

# Filtra apenas pedidos que foram entregues ('delivered') para a análise.
# Insights de Negócio: A otimização de rota só faz sentido para pedidos que já foram concluídos, garantindo que o modelo seja treinado com dados de entregas reais.
df_delivered = df[df['status_pedido'] == 'delivered'].copy()
# Remove linhas com valores nulos nas dimensões do produto (peso, volume), que são essenciais para o cálculo das restrições.
# Insights de Negócio: O tratamento de NaNs é crucial para evitar falhas em cálculos subsequentes e para garantir que apenas dados completos sejam usados na otimização da carga do caminhão.
df_delivered.dropna(subset=['peso_produto', 'comprimento_produto', 'altura_produto', 'largura_produto'], inplace=True)

# Calcula o volume do produto (comprimento * altura * largura).
# Insights de Negócio: O volume é um dos principais fatores de restrição para caminhões, junto com o peso. Incluir essa variável torna o modelo mais realista.
df_delivered['volume_produto'] = df_delivered['comprimento_produto'] * df_delivered['altura_produto'] * df_delivered[
    'largura_produto']

# Agrupa os pedidos por cidade e estado para calcular o peso e volume total por cidade.
# Insights de Negócio: Para o roteamento, o que importa é a carga total por destino, e não por item. Agrupar os dados por cidade simplifica o problema e o torna mais eficiente.
df_agrupado_cidade = df_delivered.groupby(['estado_cliente', 'cidade_cliente']).agg(
    peso_total=('peso_produto', 'sum'),
    volume_total=('volume_produto', 'sum')
).reset_index()

# Define a capacidade do caminhão (simulação)
# Insights de Negócio: Estas são as restrições de um problema real de VRP (Vehicle Routing Problem). O modelo precisa respeitar as limitações de peso e volume para ser útil.
capacidade_caminhao_peso = 100000  # em gramas (100 kg)
capacidade_caminhao_volume = 15000000  # em cm³

# Seleciona cidades aleatórias que se encaixem na capacidade do caminhão.
# Insights de Negócio: Isso simula o carregamento diário de um caminhão com pedidos que "cabem" nele, fornecendo uma base de dados realista para o problema de otimização.
pontos_da_rota_df = pd.DataFrame(columns=['estado_cliente', 'cidade_cliente', 'peso_total', 'volume_total'])
peso_acumulado = 0
volume_acumulado = 0
cidades_para_adicionar = df_agrupado_cidade.sample(frac=1).reset_index(drop=True)

for index, row in cidades_para_adicionar.iterrows():
    if peso_acumulado + row['peso_total'] <= capacidade_caminhao_peso and volume_acumulado + row[
        'volume_total'] <= capacidade_caminhao_volume:
        pontos_da_rota_df = pd.concat([pontos_da_rota_df, pd.DataFrame([row])], ignore_index=True)
        peso_acumulado += row['peso_total']
        volume_acumulado += row['volume_total']
    if len(pontos_da_rota_df) >= 20:  # Limita a 20 cidades para otimizar o tempo de execução
        break

# Cria a lista de cidades para a rota.
pontos_da_rota = pontos_da_rota_df['cidade_cliente'].str.lower().str.strip().tolist()
# Adiciona 'sao paulo' como ponto de partida (base da empresa).
if 'sao paulo' not in pontos_da_rota:
    pontos_da_rota.insert(0, 'sao paulo')

# --- 3. Geocodificação (usando dados reais para capitais e simulados para outros) ---
# Dicionário com coordenadas reais para as principais cidades.
# Insights de Negócio: Em um projeto real, essa etapa usaria uma API de geocodificação para obter as coordenadas exatas. Aqui, a simulação e o uso de dados reais para capitais tornam a análise mais robusta.
cidades_coords_reais = {
    'sao paulo': (-23.55052, -46.633309), 'rio de janeiro': (-22.9068, -43.1729),
    'belo horizonte': (-19.9167, -43.9345), 'vitoria': (-20.3150, -40.3128),
    'curitiba': (-25.4284, -49.2733), 'florianopolis': (-27.5935, -48.5585),
    'porto alegre': (-30.0346, -51.2177), 'salvador': (-12.9714, -38.5014),
    'brasilia': (-15.7797, -47.9297), 'goiania': (-16.6869, -49.2643),
    'fortaleza': (-3.7319, -38.5267), 'recife': (-8.0578, -34.8829),
    'manaus': (-3.1190, -60.0217),
    # Adicionando cidades de SP para uma simulação mais realista
    'campinas': (-22.9056, -47.0608), 'ribeirao preto': (-21.1764, -47.8183),
    'sorocaba': (-23.5029, -47.4589), 'jundiai': (-23.1866, -46.883),
    'piracicaba': (-22.723, -47.6534)
}
# Cria o dicionário de coordenadas da rota, usando dados reais quando disponíveis, e simula o restante.
# Insights de Negócio: A simulação de coordenadas é um tratamento de dado necessário quando não se tem acesso a APIs, permitindo que a análise prossiga.
cidades_coords = {cidade: cidades_coords_reais.get(cidade, (random.uniform(-35, 5), random.uniform(-75, -35))) for
                  cidade in pontos_da_rota}


# --- 4. Funções Essenciais do Algoritmo Genético ---
# Funções para cálculo de distâncias
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância Haversine (linha reta) entre dois pontos (em km)."""
    R = 6371
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def simular_distancia_real_estrada(lat1, lon1, lat2, lon2):
    """
    Simula a distância real de estrada usando um fator de correção.
    Insights de Negócio: A distância em linha reta não é um bom indicador para a logística de caminhão.
    Um fator de 1.25 a 1.40 é comum para converter distância em linha reta para distância de estrada, o que torna a estimativa de custos e tempo mais precisa.
    """
    fator_estrada = 1.25
    distancia_haversine = haversine_distance(lat1, lon1, lat2, lon2)
    return distancia_haversine * fator_estrada


def criar_matriz_distancia(coords):
    """Cria uma matriz de distâncias usando a fórmula de Haversine."""
    tamanho = len(coords)
    distancia_matrix = np.zeros((tamanho, tamanho))
    pontos_lista = list(coords.keys())
    for i in range(tamanho):
        for j in range(tamanho):
            if i != j:
                ponto1 = coords[pontos_lista[i]]
                ponto2 = coords[pontos_lista[j]]
                # Usa a nova função que simula a distância de estrada
                distancia_matrix[i, j] = simular_distancia_real_estrada(ponto1[0], ponto1[1], ponto2[0], ponto2[1])
    return distancia_matrix, pontos_lista


# Funções principais do Algoritmo Genético
def criar_rota_aleatoria(pontos_lista):
    """Cria uma rota aleatória, partindo do ponto 0 (São Paulo)."""
    rota = pontos_lista[1:]
    random.shuffle(rota)
    return [pontos_lista[0]] + rota


def funcao_fitness(rota, distancia_matrix, pontos_lista):
    """
    Função de fitness que avalia a qualidade da rota.
    Insights de Negócio: A fitness é o inverso da distância total da rota. O algoritmo
    irá buscar o maior valor de fitness, o que significa a menor distância, traduzindo o problema em uma meta clara de otimização.
    """
    distancia_total = 0
    mapa_indices = {cidade: i for i, cidade in enumerate(pontos_lista)}
    for i in range(len(rota) - 1):
        cidade_atual_idx = mapa_indices[rota[i]]
        proxima_cidade_idx = mapa_indices[rota[i + 1]]
        distancia_total += distancia_matrix[cidade_atual_idx, proxima_cidade_idx]

    distancia_total += distancia_matrix[mapa_indices[rota[-1]], mapa_indices[rota[0]]]
    return 1 / distancia_total


def crossover_ciclo(parent1, parent2):
    """Operador de crossover por ciclo para combinar rotas."""
    size = len(parent1)
    child = [None] * size
    map_p1 = {city: i for i, city in enumerate(parent1)}

    if size <= 2:
        return parent1

    cycle_start = random.randint(1, size - 1)

    current_idx = cycle_start
    while True:
        child[current_idx] = parent1[current_idx]
        next_city = parent2[current_idx]
        current_idx = map_p1[next_city]
        if current_idx == cycle_start:
            break

    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]

    capital = parent1[0]
    child_sem_capital = [c for c in child if c != capital]
    return [capital] + child_sem_capital


def mutacao_inversao(rota):
    """Operador de mutação por inversão para introduzir diversidade."""
    # Garante que a rota tem pelo menos 3 cidades para poder mutar
    if len(rota) <= 3:
        return rota

    i, j = sorted(random.sample(range(1, len(rota)), 2))
    rota[i:j + 1] = rota[i:j + 1][::-1]
    return rota


# --- 5. Estrutura Principal do Algoritmo ---
def algoritmo_genetico(pontos_coords, tamanho_populacao=200, geracoes=1000, taxa_mutacao=0.05):
    """Função principal do Algoritmo Genético."""
    distancia_matrix, pontos_lista = criar_matriz_distancia(pontos_coords)

    if len(pontos_lista) <= 2:
        print("Erro: A rota precisa de mais de uma cidade de destino.")
        return pontos_lista, 0, 0

    start_time = datetime.now()
    populacao = [criar_rota_aleatoria(pontos_lista) for _ in range(tamanho_populacao)]
    melhor_rota = None
    melhor_fitness = -1

    print("Iniciando o Algoritmo Genético...")
    # Itera sobre as gerações para encontrar a melhor rota.
    for geracao in range(geracoes):
        fitness_populacao = [(funcao_fitness(rota, distancia_matrix, pontos_lista), rota) for rota in populacao]
        fitness_populacao.sort(key=lambda x: x[0], reverse=True)

        if fitness_populacao[0][0] > melhor_fitness:
            melhor_fitness = fitness_populacao[0][0]
            melhor_rota = fitness_populacao[0][1]

        elite_size = int(tamanho_populacao * 0.1)
        proxima_geracao = [rota for _, rota in fitness_populacao[:elite_size]]

        while len(proxima_geracao) < tamanho_populacao:
            pai1 = random.choice(proxima_geracao)
            pai2 = random.choice(proxima_geracao)
            filho = crossover_ciclo(pai1, pai2)

            if random.random() < taxa_mutacao:
                filho = mutacao_inversao(filho)
            proxima_geracao.append(filho)

        populacao = proxima_geracao

        if (geracao + 1) % 100 == 0:
            melhor_distancia = 1 / melhor_fitness
            print(f"Geração {geracao + 1}: Melhor Distância Encontrada = {melhor_distancia:.2f} km")

    end_time = datetime.now()
    tempo_execucao = end_time - start_time
    melhor_distancia = 1 / melhor_fitness

    return melhor_rota, melhor_distancia, tempo_execucao


def plotar_rota(coords, rota, distancia_total, peso_total, volume_total):
    """
    Plota a melhor rota encontrada no contexto do mapa do Brasil (simplificado).
    Também exibe a distância e os dados da carga.
    Insights de Negócio: A visualização é um insight crucial para o gestor,
    pois apresenta de forma clara e intuitiva o resultado da otimização, facilitando a tomada de decisão.
    """
    plt.figure(figsize=(12, 8))

    # Dicionário com coordenadas das capitais para contextualizar o mapa
    estados_coords = {
        'AC': (-9.97, -67.81), 'AL': (-9.66, -35.73), 'AP': (0.03, -51.05),
        'AM': (-3.11, -60.02), 'BA': (-12.97, -38.50), 'CE': (-3.73, -38.52),
        'DF': (-15.79, -47.88), 'ES': (-20.3150, -40.3128), 'GO': (-16.68, -49.25),
        'MA': (-2.53, -44.30), 'MG': (-19.9167, -43.9345), 'MS': (-20.44, -54.65),
        'MT': (-15.59, -56.09), 'PA': (-1.45, -48.50), 'PB': (-7.11, -34.86),
        'PE': (-8.05, -34.88), 'PI': (-5.09, -42.80), 'PR': (-25.4284, -49.2733),
        'RJ': (-22.9068, -43.1729), 'RN': (-5.79, -35.20), 'RO': (-8.76, -63.90),
        'RR': (2.81, -60.75), 'RS': (-30.0346, -51.2177), 'SC': (-27.5935, -48.5585),
        'SE': (-10.91, -37.07), 'SP': (-23.55052, -46.633309), 'TO': (-10.18, -48.33)
    }

    for estado, coord in estados_coords.items():
        plt.text(coord[1], coord[0], estado, fontsize=9, ha='center', va='center', color='gray', alpha=0.5)

    # Plota a rota. Iteramos pela rota para garantir a ordem correta.
    for i, nome in enumerate(rota):
        coords_vals = coords[nome]
        if i == 0:
            # Ponto de Partida
            plt.plot(coords_vals[1], coords_vals[0], 's', markersize=12, color='green', label='Partida')
            plt.text(coords_vals[1], coords_vals[0], ' Partida (SP)', fontsize=12, verticalalignment='bottom',
                     fontweight='bold', color='darkgreen')
        elif i == len(rota) - 1:
            # Ponto de Chegada (última cidade antes de voltar para a base)
            plt.plot(coords_vals[1], coords_vals[0], 'o', markersize=12, color='red', label='Chegada')
            plt.text(coords_vals[1], coords_vals[0], f' {nome} - Chegada', fontsize=12, verticalalignment='bottom',
                     fontweight='bold', color='darkred')
        else:
            # Demais pontos de entrega
            plt.plot(coords_vals[1], coords_vals[0], 'o', markersize=10, color='blue')
            plt.text(coords_vals[1], coords_vals[0], f' {nome}', fontsize=12, verticalalignment='bottom')

    rota_ordenada = [coords[c] for c in rota]
    x = [p[1] for p in rota_ordenada]
    y = [p[0] for p in rota_ordenada]

    x.append(x[0])
    y.append(y[0])

    plt.plot(x, y, 'r--', linewidth=2, alpha=0.7)

    # Sinaliza o valor total da rota
    plt.text(
        0.05, 0.95,
        f"Distância Total: {distancia_total:.2f} km",
        transform=plt.gca().transAxes,
        fontsize=14,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8)
    )
    plt.text(
        0.05, 0.90,
        f"Peso Total: {peso_total / 1000:.2f} kg\nVolume Total: {volume_total:.2f} cm³",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8)
    )

    # Adiciona a legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title('Melhor Rota Encontrada por Algoritmo Genético', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # O gráfico gerado está sendo salvo com o nome 'melhor_rota.png'
    plt.savefig('melhor_rota.png')
    plt.show()


melhor_rota_encontrada, melhor_distancia_encontrada, tempo_execucao = algoritmo_genetico(cidades_coords)
peso_total_rota = pontos_da_rota_df['peso_total'].sum()
volume_total_rota = pontos_da_rota_df['volume_total'].sum()

# Formata a saída para o terminal e para o arquivo de texto
output_string = (
    "--- Análise Completa da Rota ---\n"
    f"Melhor Rota Encontrada:\n"
    f"{' -> '.join(melhor_rota_encontrada)}\n"
    f"Distância Total da Melhor Rota: {melhor_distancia_encontrada:.2f} km\n"
    f"Peso Total da Carga: {peso_total_rota / 1000:.2f} kg\n"
    f"Volume Total da Carga: {volume_total_rota:.2f} cm³\n"
    f"Tempo de Execução do Algoritmo: {str(tempo_execucao)}\n"
)

# Imprime no terminal
print(output_string)

# Salva em um arquivo de texto
with open('resultado_TargetCaminhoRota.txt', 'w') as f:
    f.write(output_string)

plotar_rota(cidades_coords, melhor_rota_encontrada, melhor_distancia_encontrada, peso_total_rota, volume_total_rota)
