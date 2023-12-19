
from django.urls import path
from . import views

urlpatterns = [
    path('',views.hello,name='hello'),
    path('index/',views.index,name='test'),
]