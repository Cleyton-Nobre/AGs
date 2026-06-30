import numpy as np
import pandas as pd
import Iniciar_dados as init_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Users = []
PAs = []
MAX_PAS = 30
TOTAL_USERS = 495

import numpy as np
import copy

def gerar_sol_inicial(k=10):
    # Load data from file
    dados = np.loadtxt('clientes.txt', delimiter=',')

    # Extract X and Y coordinates
    X = dados[:, 0]
    Y = dados[:, 1]

    dados2 =  [(x[0], x[1]) for x in dados]

    # Apply K-Means algorithm
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dados2)
    Clusters_kMeans = kmeans.predict(dados2)
    Centroids_kMeans = kmeans.cluster_centers_

    # Convert dados2 to a 2D array
    dados2_array = np.array(dados2)

    centroides_arredondadas = np.round(Centroids_kMeans / 5) * 5

    posicoes = []
    for x in centroides_arredondadas:
        posicoes.append(int((x[0]/5)*80 + (x[1]/5)))

    vetor = np.zeros(6400)

    for posicao in posicoes:
        vetor[posicao] = 1

    return vetor

# def gerar_resultados(vetor, PAs, Users):
#     PAs_ativados = ativar_PAs(copy.deepcopy(PAs), vetor)  # shallow copy
#     users = copy.deepcopy(Users)  # shallow copy

#     n_UserConectados = 0
#     total_distance = 0

#     for user in users:
#         if n_UserConectados > 0.95*TOTAL_USERS:
#             break

#         for i in range (6400):
#             if not user.user_atendido and float(user.distancias_pas[i]) <= PAs_ativados[user.indices[i]].raio and PAs_ativados[user.indices[i]].banda_disponivel >= user.demandaRede and PAs_ativados[user.indices[i]].PA_ativado:
#                 PAs_ativados[user.indices[i]].usuarios_atendidos.append(user)
#                 user.user_atendido = True
#                 PAs_ativados[user.indices[i]].banda_disponivel -= user.demandaRede
#                 user.PA_conectado = PAs_ativados[user.indices[i]].coordenadas
#                 total_distance += float(user.distancias_pas[i])
#                 n_UserConectados += 1
#                 break
    
#     PAs_ativados__ = []

#     for elemento in PAs_ativados:
#         if elemento.PA_ativado:
#             PAs_ativados__.append(elemento)

#     return PAs_ativados__, users   


def gerar_resultados(vetor, PAs, users_originais):
    # Ativa os PAs conforme o vetor de ativação
    PAs_ativados = ativar_PAs(copy.deepcopy(PAs), vetor)
    users = copy.deepcopy(users_originais)

    num_conectados = 0
    total_distancia = 0

    limite_conexao = int(0.98 * TOTAL_USERS)

    for user in users:
        if num_conectados >= limite_conexao:
            break

        for idx in range(6400):  # Ou substitua por len(user.indices) se aplicável
            i_pa = user.indices[idx]
            pa = PAs_ativados[i_pa]
            distancia = float(user.distancias_pas[idx])

            if (
                not user.user_atendido and
                pa.PA_ativado and
                distancia <= pa.raio and
                pa.banda_disponivel >= user.demandaRede
            ):
                pa.usuarios_atendidos.append(user)
                pa.banda_disponivel -= user.demandaRede

                user.user_atendido = True
                user.PA_conectado = pa.coordenadas
                total_distancia += distancia
                num_conectados += 1
                break

    # Filtra apenas os PAs ativados
    PAs_ativos = [pa for pa in PAs_ativados if pa.PA_ativado]

    return PAs_ativos, users


def ativar_PAs(Pontos_acesso, vetor):
    count = 0
    for i in range(len(vetor)):
        if vetor[i] == 1:
            Pontos_acesso[i].PA_ativado = True
            count += 1

        elif count == MAX_PAS:
            break
        
    return Pontos_acesso

def visulizar_solucao(PAs_,users_):
    clientes=init_data.ler_dados_csv()
    clusters = []
    centroids = []

    # Supondo que cada 'v' em clientes seja uma tupla ou lista, onde v[0] também é uma tupla/lista com duas coordenadas
    t = [v[0][0] for v in clientes]
    t1 = [v[0][1] for v in clientes]

    # Criar DataFrame com colunas nomeadas
    dfo = pd.DataFrame({'X': t, 'Y': t1})

    # Construir lista de centróides com índices
    for i, pa in enumerate(PAs_):
        if pa.PA_ativado:
            centroids.append((np.array(pa.coordenadas), i))

    # Associar usuários atendidos aos centróides
    for user in users_:
        if user.user_atendido:
            user_coords = np.array(user.coordenadas)
            pa_coords = np.array(user.PA_conectado)
            for coords, idx in centroids:
                if np.all(coords == pa_coords):
                    clusters.append(np.append(user_coords, idx))

    # Converter clusters para DataFrame
    df = pd.DataFrame(clusters, columns=["X", "Y", "Cluster"])

    df_combinado = pd.concat([df, dfo])
    df_s= df_combinado.drop_duplicates(subset=['X', 'Y'], keep=False)

    # Extrair apenas coordenadas dos centróides para plotagem
    centroids_coords = np.array([c[0] for c in centroids])

    # Plotar os pontos e os centróides
    plt.figure(figsize=(14, 7))
    plt.scatter(df_s['X'], df_s['Y'], alpha=0.9, c="black", label='Clientes não Atendidos', s=20)
    plt.scatter(df['X'], df['Y'], c=df['Cluster'], cmap='viridis', alpha=0.5, label='Clientes Atendidos', s=20)
    plt.scatter(centroids_coords[:, 0], centroids_coords[:, 1], color='red', marker='o', s=50, label='Centroides')

    # Desenhar linhas ligando cada ponto ao seu centróide
    for _, row in df.iterrows():
        cluster_idx = int(row['Cluster'])
        centroide = centroids_coords[cluster_idx]
        plt.plot([row['X'], centroide[0]], [row['Y'], centroide[1]], '--', color='gray', linewidth=1, alpha=0.8)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("KMeans Clustering dos Clientes com Ligações aos Centroides")
    plt.legend()
    plt.grid(True)
    plt.show()
