"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from predict import views
from django.conf import settings
from django.conf.urls.static import static

# router = routers.DefaultRouter()
# router.register(r'predictions', views.PredictionView, 'predict')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/single-pair/', views.single_pair_predict),
    path('api/many-pair/', views.many_pair_predict),
    path('api/all-pair/', views.all_pair_predict),
    path('api/predict/', views.predict),
    path('api/test', views.test_append),
    path('api/position', views.get_queue_pos),
    path('api/process', views.process_jobs),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)