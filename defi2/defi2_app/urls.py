# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name='index'),
     path('/import_excel',views.import_excel, name='import_excel'),
      path('graph/', views.graph_page, name='graph_page'),
      
]
