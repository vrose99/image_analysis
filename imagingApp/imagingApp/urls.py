"""
URL configuration for imagingApp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path
from dashboard.views import *
from django.conf import settings
from django.views.static import serve


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', DashboardView.as_view(), name="dashboard"),
    path('highpass/', HighView.as_view(), name="highpass"),
    path('lowpass/', LowView.as_view(), name="lowpass"),
    path('intro/', IntroView.as_view(), name="intro"),
    path('other/', OtherView.as_view(), name="other"),
    path('bib/',BibView.as_view(), name="bib")
    
]


