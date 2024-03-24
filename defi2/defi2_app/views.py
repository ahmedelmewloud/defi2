from django.http import JsonResponse
from django.shortcuts import render,redirect
import pandas as pd
from django.core.files.storage import default_storage
from django.conf import settings
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import base64
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from geopy.distance import geodesic
import pandas as pd
import io
import urllib

def index(request):
    if request.method == 'POST':
        choix = request.POST.get('choix')
        if choix == 'app':
            return redirect('graph_page')
        elif choix == 'aco':
            return redirect('graphe_aco2')
    return render(request, 'index.html')
# df = 0
def import_excel(request):
    if request.method == 'POST':
        excel_file = request.FILES.get('excel_file')
        if excel_file and excel_file.name.endswith('.xlsx'):
            # Enregistrer le fichier dans un dossier spécifié
            file_path = default_storage.save(settings.MEDIA_ROOT / 'uploads' / excel_file.name, excel_file)
            
            # Traiter les données Excel ici
            # Par exemple, enregistrer les données dans la base de données
            
            # Rediriger l'utilisateur vers une page de succès
            return redirect('index')
        else:
            return render(request, 'h.html', {'error_message': 'Le fichier doit être au format Excel (.xlsx)'})
    else:
        return render(request, 'h.html')



def graph_page(request):
    # Chemin du fichier Excel
    file_path = r"C:\med\Cordonnees_GPS.xlsx"
    
    # Lecture des données du fichier Excel
    df = pd.read_excel(file_path)
    
    # Initialisation de la matrice des distances
    n = len(df)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                coord1 = (df.iloc[i]['Latitude'], df.iloc[i]['Longitude'])
                coord2 = (df.iloc[j]['Latitude'], df.iloc[j]['Longitude'])
                distances[i][j] = geodesic(coord1, coord2).km
    
    # Trouver l'index de Nouakchott
    start_index = df.index[df['Ville'].str.strip() == 'Nouakchott'].tolist()[0]
    
    # Heuristique du plus proche voisin pour le TSP
    def nearest_neighbor_tsp(start_index, distances):
        path = [start_index]
        n = len(distances)
        mask = np.zeros(n, dtype=bool)
        mask[start_index] = True

        for _ in range(n - 1):
            last = path[-1]
            next_indices = np.argsort(distances[last])
            for next_index in next_indices:
                if not mask[next_index]:
                    path.append(next_index)
                    mask[next_index] = True
                    break

        path.append(start_index)  # Retour à la ville de départ pour fermer le circuit
        return path
    
    # Calculer le chemin
    path_indices = nearest_neighbor_tsp(start_index, distances)
    path_villes = [df.iloc[i]['Ville'] for i in path_indices]
    
    # Préparation des positions des villes pour le tracé
    pos = {df.iloc[i]['Ville']: (df.iloc[i]['Longitude'] - 0.5, df.iloc[i]['Latitude']) for i in path_indices}
    
    # Création du graphe pour le tracé
    G = nx.Graph()
    for i in range(len(path_indices) - 1):
        G.add_edge(df.iloc[path_indices[i]]['Ville'], df.iloc[path_indices[i + 1]]['Ville'])
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=2, node_color='red')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=[(path_villes[i], path_villes[i+1]) for i in range(len(path_villes)-2)], edge_color='blue', arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=[(path_villes[-2], path_villes[-1])], edge_color='red', arrows=True, style='dashed')

    # Ajout de la légende
    plt.text(0.5, 1.05, 'Rouge: Retour', transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, 1.02, 'Bleu: Aller', transform=plt.gca().transAxes, ha='center')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    image_base64 = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'graph_page.html', {'image_base64': image_base64})








def graphe_aco2(request):
    # Chargement des données
    file_path = r"C:\med\Cordonnees_GPS.xlsx"  # Mettez ici le chemin correct vers votre fichier Excel
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
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="blue")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="gray")
    plt.title('Tournée minimale en Mauritanie avec ACO')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('off')

    # Enregistrer l'image générée dans un objet BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convertir l'image en une chaîne Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Fermer la figure de Matplotlib pour libérer la mémoire
    plt.close()

    # Passer la chaîne Base64 à la page HTML pour l'affichage
    return render(request, 't.html', {'image_base64': image_base64})




