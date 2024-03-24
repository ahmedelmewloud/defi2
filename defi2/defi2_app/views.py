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

from geopy.distance import geodesic
import pandas as pd
import io
import urllib

def index(request):
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

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    image_base64 = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'graph_page.html', {'image_base64': image_base64})
