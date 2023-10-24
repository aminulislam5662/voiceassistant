


from django.urls import path

from assistantapp import views 
from .views import *

app_name = "dashboard"


urlpatterns = [
path("", views.index, name="index"),

path("api/search/bing", BingSearch.as_view()),
path("api/search/duck", Duckduckgo.as_view()),
path("api/createmodel", CreateModel.as_view()),
path("api/createmodel2", CreateModel2.as_view()),
path("api/semanticsearch", SemanticSearch.as_view()),

]