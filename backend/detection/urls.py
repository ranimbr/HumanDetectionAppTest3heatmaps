from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_image/', views.upload_image, name='upload_image'),

    # Trois URLs vidéo différentes pour trois détections différentes
    path('upload_video_density_zones/', views.detect_video1, name='upload_video_density_zones'),
    path('upload_video_stop_time/', views.detect_video2, name='upload_video_stop_time'),
    path('upload_video_product_interaction/', views.detect_video3, name='upload_video_product_interaction'),

   
]
