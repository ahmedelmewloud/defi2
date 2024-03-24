from django.shortcuts import render
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import networkx as nx


def index(request):
    return render(request, 'index.html')

def graphe(request):


    # Chargement des données
    file_path = r"C:\Users\lapto\Desktop\clone\defi2\defi2\Cordonnees_GPS.xlsx"  # Mettez ici le chemin correct vers votre fichier Excel
    df = pd.read_excel(file_path, sheet_name='Capitales_Wilaya')

    # Trouver l'index de Nouakchott
    start_index = df.index[df['Ville'] == 'Nouakchott'].tolist()[0]

    # Calcul de la matrice des distances
    n = len(df)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                coord_i = (df.iloc[i]['Latitude'], df.iloc[i]['Longitude'])
                coord_j = (df.iloc[j]['Latitude'], df.iloc[j]['Longitude'])
                distances[i][j] = geodesic(coord_i, coord_j).km

    # Implémentation simplifiée de l'ACO
    def aco_find_path(distances, start_index, num_ants=10, num_iterations=100, decay=0.5, alpha=1, beta=2):
        best_path = None
        best_length = np.inf
        best_history = []
        for iteration in range(num_iterations):
            for ant in range(num_ants):
                path = [start_index]
                while len(path) < len(distances):
                    current_city = path[-1]
                    probabilities = []
                    for city in range(len(distances)):
                        if city not in path:
                            pheromone = ((1.0 / distances[current_city][city]) ** beta)
                            probabilities.append(pheromone)
                        else:
                            probabilities.append(0)
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    next_city = np.random.choice(range(len(distances)), p=probabilities)
                    path.append(next_city)
                path_length = sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1))
                path_length += distances[path[-1], path[0]]
                if path_length < best_length:
                    best_length = path_length
                    best_path = path
                    best_history.append([df.iloc[i]['Ville'] for i in path])  # Track history of visited cities
        return best_path, best_length, best_history

    # Trouver le meilleur chemin
    best_path, best_length, best_history = aco_find_path(distances, start_index)

    # Afficher le meilleur chemin et sa longueur
    print("Meilleur chemin:", [df.iloc[i]['Ville'] for i in best_path])
    print("Longueur du chemin:", best_length)

    # Afficher l'historique des villes visitées
    print("\nHistorique des villes visitées:")
    for idx, cities in enumerate(best_history):
        print(f"Iteration {idx + 1}: {cities}")

    # Visualisation avec NetworkX
    G = nx.Graph()

    for i, row in df.iterrows():
        G.add_node(row['Ville'], pos=(row['Longitude'], row['Latitude']))

    for i in range(len(best_path) - 1):
        G.add_edge(df.iloc[best_path[i]]['Ville'], df.iloc[best_path[i + 1]]['Ville'])

    # Pour fermer le chemin
    G.add_edge(df.iloc[best_path[-1]]['Ville'], df.iloc[best_path[0]]['Ville'])

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="blue")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="gray")
    plt.title('Tournée minimale en Mauritanie avec ACO')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('off')
    plt.show()
    return render(request,'t.html')
