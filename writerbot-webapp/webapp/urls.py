from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home),
    path('write/', views.write),
    path('about/', views.about),
    path('team/', views.team)
]
