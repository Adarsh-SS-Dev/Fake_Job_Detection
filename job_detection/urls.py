from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_fake_job_view, name='predict_fake_job'),
]
