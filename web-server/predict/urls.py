from django.urls import path
from . import views

urlpatterns = [
    path('api/predict/', views.predict),
    path('api/process', views.process_jobs),
    path('api/position/<uuid:id>/', views.get_pos),
]