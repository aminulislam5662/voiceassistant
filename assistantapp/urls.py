


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
path("api/summary", SummaryApi.as_view()),

]

# mkdir filename
# sudo apt install python3-
# python3 -m venv myenv
# source myenv/bin/activate
# cd filename
# cd ..
# sudo python3 manage.py runserver 0.0.0.0:80
# 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
#tmux kill-session
# git pull https://github.com/aminulislam5662/voiceassistant.git
# sudo kill -9 $(sudo lsof -t -i:80)