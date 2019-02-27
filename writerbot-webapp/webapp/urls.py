from django.urls import path, include
from django.conf import settings
from . import views

urlpatterns = [
    path('', views.home),
    path('write/', views.write),
    path('about/', views.about),
    path('team/', views.team),
    path('logout/', views.logout),
    path('error/', views.error),
    path('new_story', views.newStory),
    path('load_story/<str:id>', views.loadStory),
    path('delete_story/<str:id>', views.deleteStory)
]
