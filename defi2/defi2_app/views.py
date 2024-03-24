file_path = r"C:\Users\LAPTOP\Desktop\defi2\defi2\defi2_app\Cordonnees_GPS.xlsx"

from django.http import JsonResponse
from django.shortcuts import render,redirect,HttpResponse
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
import matplotlib.animation as animation

def index(request):
    if request.method == 'POST':
        choix = request.POST.get('choix')
        if choix == 'app':
            return redirect('graph_page')
        elif choix == 'aco':
            return redirect('graphe_aco2')
        elif choix == 'aco_graph':
             return redirect('graphe_aco')
        elif choix == 'app_graph':
            return redirect('graph_p')
        elif choix == 'Apr_anim':
            return redirect('graph_approx_anim')
    return render(request, 'index.html')

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

# def graph_page(request):
#     files = os.listdir()
#     # Filter Excel files
#     excel_files = [file for file in files if file.endswith('.xlsx')]
#     # Check if there's exactly one Excel file in the directory
#     if len(excel_files) == 1:
#         # Read the Excel file
#         excel_file_path = excel_files[0]

#         # Lecture des données du fichier Excel
#         df = pd.read_excel(excel_file_path)

#         # Préparation des données pour le tracé
#         x = df['Longitude']
#         y = df['Latitude']
#         labels = df['Ville']

#         # Création du graphe sous forme de scatter plot avec les lignes reliant les points
#         plt.figure(figsize=(12, 8))

#         # Tracé des points
#         plt.scatter(x, y, color='lightblue', edgecolors='black')
#         for i, label in enumerate(labels):
#             plt.text(x[i], y[i], label, fontsize=8, ha='right')

#         # Tracé des lignes reliant les points
#         for i in range(len(df)):
#             plt.plot([x[i]], [y[i]], marker='o', markersize=5, color='red')
#             if i < len(df) - 1:
#                 plt.plot([x[i], x[i+1]], [y[i], y[i+1]], linestyle='-', color='blue')

#         # Configuration des axes et du titre
#         plt.xlabel('Longitude')
#         plt.ylabel('Latitude')
#         plt.title('Villes - Coordonnées GPS')

#         # Ajout de la grille
#         plt.grid(True)

#         # Modification de la couleur de l'arrière-plan
#         plt.gca().set_facecolor('white')

#         # Conversion du graphe en image et encodage en base64
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png')
#         buffer.seek(0)
#         image_png = buffer.getvalue()
#         buffer.close()
#         image_base64 = base64.b64encode(image_png).decode('utf-8')

#         return render(request, 'Approx_graph.html', {'image_base64': image_base64})
#     return HttpResponse("error")




def graph_page(request):
  
    
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
    pos = {df.iloc[i]['Ville']: (df.iloc[i]['Longitude'], df.iloc[i]['Latitude']) for i in path_indices}

    # Création du graphe pour le tracé
    G = nx.Graph()
    for i in range(len(path_indices) - 1):
        G.add_edge(df.iloc[path_indices[i]]['Ville'], df.iloc[path_indices[i + 1]]['Ville'])

    plt.figure(figsize=(12, 8))
    # Nuage de points pour les villes
    for ville, (x, y) in pos.items():
        plt.scatter(x, y, color='lightblue', edgecolor='black')
        plt.text(x, y, ville, fontsize=9, ha='right', va='bottom')

    # Tracer les lignes reliant les points
    for i in range(len(path_villes) - 1):
        ville1 = path_villes[i]
        ville2 = path_villes[i + 1]
        plt.plot([pos[ville1][0], pos[ville2][0]], [pos[ville1][1], pos[ville2][1]], color='blue')

    # Ajouter une flèche indiquant le départ de Nouakchott vers Rosso
    plt.annotate('', xy=pos['Rosso'], xytext=pos['Nouakchott'], arrowprops=dict(facecolor='red', arrowstyle='->'))

    plt.title('Nuage de points avec lignes reliant les villes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    image_base64 = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'Approx_graph.html', {'image_base64': image_base64})



from django.views.decorators.cache import cache_page
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from geopy.distance import geodesic

@cache_page(60 * 15)  # Cache for 15 minutes (in seconds)
def graphe_aco2(request):
    # Chargement des données
 # Mettez ici le chemin correct vers votre fichier Excel
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

    # Création du repère pour les points
    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé des points avec les noms des villes
    for i, city in df.iterrows():
        ax.scatter(city['Longitude'], city['Latitude'], s=100, color='red', edgecolors='black', alpha=0.7)
        ax.text(city['Longitude'], city['Latitude'], city['Ville'], fontsize=8, ha='right')

    # Tracé du trajet avec indication du sens de départ et d'arrivée
    for i in range(len(best_path) - 1):
        start_city = df.iloc[best_path[i]]['Ville']
        end_city = df.iloc[best_path[i + 1]]['Ville']
        ax.plot([df.iloc[best_path[i]]['Longitude'], df.iloc[best_path[i + 1]]['Longitude']],
                [df.iloc[best_path[i]]['Latitude'], df.iloc[best_path[i + 1]]['Latitude']],
                color='blue', linewidth=2, alpha=0.7)
        ax.text(df.iloc[best_path[i]]['Longitude'], df.iloc[best_path[i]]['Latitude'], start_city, fontsize=8, ha='right')
        ax.text(df.iloc[best_path[i + 1]]['Longitude'], df.iloc[best_path[i + 1]]['Latitude'], end_city, fontsize=8, ha='right')

    # Pour relier la première et la dernière ville du trajet
    start_city = df.iloc[best_path[0]]['Ville']
    end_city = df.iloc[best_path[-1]]['Ville']
    ax.plot([df.iloc[best_path[0]]['Longitude'], df.iloc[best_path[-1]]['Longitude']],
            [df.iloc[best_path[0]]['Latitude'], df.iloc[best_path[-1]]['Latitude']],
            color='blue', linewidth=2, alpha=0.7)
    ax.text(df.iloc[best_path[0]]['Longitude'], df.iloc[best_path[0]]['Latitude'], start_city, fontsize=8, ha='right')
    ax.text(df.iloc[best_path[-1]]['Longitude'], df.iloc[best_path[-1]]['Latitude'], end_city, fontsize=8, ha='right')

    # Ajout de la flèche pour indiquer le sens de départ
    ax.annotate('', xy=(df.iloc[best_path[1]]['Longitude'], df.iloc[best_path[1]]['Latitude']),
            xytext=(df.iloc[best_path[0]]['Longitude'], df.iloc[best_path[0]]['Latitude']),
            arrowprops=dict(facecolor='green', arrowstyle='-|>', connectionstyle="arc3,rad=.5"))

    # Configuration des axes
    ax.set_title('Tournée minimale en Mauritanie avec ACO')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)

    # Enregistrer l'image générée dans un objet BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convertir l'image en une chaîne Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Fermer la figure de Matplotlib pour libérer la mémoire
    plt.close()

    # Passer la chaîne Base64 à la page HTML pour l'affichage
    return render(request, 'ACO_REP.html', {'image_base64': image_base64})





















def graphe_aco(request):
    # Chargement des données
     # Mettez ici le chemin correct vers votre fichier Excel
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
    return render(request, 'ACO_GRAPH.html', {'image_base64': image_base64})
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from geopy.distance import geodesic
import os








def graph_p(request):
    # Chemin du fichier Excel
    # file_path = r"C:\Users\LAPTOP\Desktop\defi2\defi2\Cordonnees_GPS.xlsx"

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
    pos = {df.iloc[i]['Ville']: (df.iloc[i]['Longitude'], df.iloc[i]['Latitude']) for i in path_indices}

    # Création du graphe pour le tracé
    G = nx.Graph()
    for i in range(len(path_indices) - 1):
        G.add_edge(df.iloc[path_indices[i]]['Ville'], df.iloc[path_indices[i + 1]]['Ville'])

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=[(path_villes[i], path_villes[i+1]) for i in range(len(path_villes)-2)], edge_color='blue', arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=[(path_villes[-2], path_villes[-1])], edge_color='red', arrows=True, style='dashed')

    # Ajouter la légende
    plt.text(0.02, 0.98, 'Rouge indique Retour', color='red', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.02, 0.94, 'Bleu indique Aller', color='blue', fontsize=12, transform=plt.gca().transAxes)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    image_base64 = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'Approx_rep.html', {'image_base64': image_base64})

import tempfile

import os

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geopy.distance import geodesic
import io
import base64

def graph_approx_anim(request):
    # Chemin du fichier Excel
    # file_path = r"C:\Users\LAPTOP\Desktop\defi2\defi2\Cordonnees_GPS.xlsx"

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
    pos = {df.iloc[i]['Ville']: (df.iloc[i]['Longitude'], df.iloc[i]['Latitude']) for i in path_indices}

    # Création du graphe pour le tracé
    G = nx.Graph()
    for i in range(len(path_indices) - 1):
        G.add_edge(df.iloc[path_indices[i]]['Ville'], df.iloc[path_indices[i + 1]]['Ville'])

    fig, ax = plt.subplots(figsize=(12, 8))
    # Nuage de points pour les villes
    for ville, (x, y) in pos.items():
        ax.scatter(x, y, color='lightblue', edgecolor='black')
        ax.text(x, y, ville, fontsize=9, ha='right', va='bottom')

    # Tracer les lignes reliant les points
    line, = ax.plot([], [], color='blue')

    # Ajouter une flèche indiquant le départ de Nouakchott vers Rosso
    ax.annotate('', xy=pos['Rosso'], xytext=pos['Nouakchott'], arrowprops=dict(facecolor='red', arrowstyle='->'))

    def animate(i):
        line.set_data([pos[path_villes[i]][0], pos[path_villes[i + 1]][0]],
                      [pos[path_villes[i]][1], pos[path_villes[i + 1]][1]])

    # Définir une valeur de délai plus grande pour ralentir l'animation
    interval = 700  # 200 millisecondes entre chaque image de l'animation

    # Créer l'animation avec le délai spécifié
    anim = animation.FuncAnimation(fig, animate, frames=len(path_villes) - 1, repeat=False, interval=interval)

    # Enregistrer l'animation dans un répertoire temporaire personnalisé
    temp_dir = '/chemin/vers/votre/repertoire/temporaire'  # Spécifiez le chemin de votre répertoire temporaire personnalisé
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, 'animation.gif')
    anim.save(temp_file, writer='pillow')

    # Lire le fichier temporaire et encoder en base64
    with open(temp_file, 'rb') as f:
        image_data = f.read()

    image_base64 = base64.b64encode(image_data).decode('utf-8')

    return render(request, 'Approx_graph_animation.html', {'image_base64': image_base64})




