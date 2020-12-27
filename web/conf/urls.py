from django.contrib import admin
from django.urls import path

from faqer.views import AskView, CategoriesView


urlpatterns = [
    path('ask/', AskView.as_view()),
    path('categories/', CategoriesView.as_view())
]
