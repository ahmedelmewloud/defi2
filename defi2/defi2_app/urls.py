from django.urls import path
from . import views  # Import views module from the current directory

urlpatterns = [
    path('',views.index, name='index'),
    path('import_excel/',views.import_excel, name='import_excel'),
    path('graph_page/', views.graph_page, name='graph_page'),
    path('graph_p/', views.graph_p, name='graph_p'),
    path('graphe_aco2/',views.graphe_aco2, name='graphe_aco2'),
    path('graphe_aco/',views.graphe_aco, name='graphe_aco'),

]
