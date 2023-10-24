from django.shortcuts import render
import json
import requests
from rest_framework import status
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import Group
from django.db.models import Sum,Avg,F
from django.shortcuts import render,get_object_or_404
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from rest_framework.views import APIView
from rest_framework.response import Response

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import sys

from txtai.embeddings import Embeddings
import json
from duckduckgo_search import DDGS

import sys
from txtai.pipeline import Summary, Textractor
import torch
from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer= T5Tokenizer.from_pretrained('t5-small',legacy=False)
device=torch.device('cpu')
# Create your views here.


def T5Summarize(text):
    preprocessed_text=text.strip().replace('\n','')

    t5inputtext='summarize '+preprocessed_text

    print(len(t5inputtext.split()))

    tokenizedtext=tokenizer.encode(t5inputtext,return_tensors='pt',truncation=True).to(device)

    summaryids=model.generate(tokenizedtext,min_length=30,max_length=120)
    summary=tokenizer.decode(summaryids[0],skip_special_tokens=True)


    print(summary)
    return summary


def index(request):
    return HttpResponse("Hello World")


class BingSearch(APIView):
    def post(self,request):
        context={}
        query = request.data['query']
        encoded_query = query.replace(" ","+")

        # Configure the Bing search URL
        search_url = f"https://www.bing.com/search?q={encoded_query}"
        # search_url = f"https://duckduckgo.com/?q={encoded_query}&ia=web"

        # Send an HTTP GET request to Bing
        response = requests.get(search_url)
        # Create a list to store the search results
        search_results_list = []

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the search results page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract search results
            search_results = soup.find_all('li', class_='b_algo')
            # search_results = soup.find_all('div', class_='result')

            # Loop through the search results and add each result to the list as a dictionary

            print(len(search_results))
            for result in search_results:
                # Extract the title and snippet (description) of the search result
                title = result.find('h2').get_text()
                snippet = result.find('p').get_text()

                # Extract the URL from the <a> tag
                result_url = result.find('a')['href']
                result_dict = {
                    'url': result_url,
                    'Title': title.encode(sys.stdout.encoding, 'replace').decode(sys.stdout.encoding),
                    'Snippet': snippet.encode(sys.stdout.encoding, 'replace').decode(sys.stdout.encoding),
                }

                search_results_list.append(result_dict)
            context={
                    "status":True,
                    "query":query,
                    "data":search_results_list,
                    }
            print(context)

        else:
            context={
                "status":False,
                "query":query,
                "data":search_results_list,
            }
            print(f"Error: Unable to fetch search results. Status code {response.status_code}")
        return Response(context)

embeddings = Embeddings({
    "path": "sentence-transformers/all-MiniLM-L6-v2"
})

def bingsearch(slug):
    fulltext=""
    encoded_query = slug.replace(" ","+")
    search_url = f"https://www.bing.com/search?q={encoded_query}"
    response = requests.get(search_url)
        # Create a list to store the search results
    text_list = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('li', class_='b_algo')
        for result in search_results:
                # Extract the title and snippet (description) of the search result
            title = result.find('h2').get_text()
            snippet = result.find('p').get_text()
            text_list.append(title.encode(sys.stdout.encoding, 'replace').decode(sys.stdout.encoding))
            text_list.append("\n")
            text_list.append(snippet.encode(sys.stdout.encoding, 'replace').decode(sys.stdout.encoding))
            text_list.append("\n")
                # Extract the URL from the <a> tag
            result_url = result.find('a')['href']
            result_dict = {
                    'url': result_url
                }
        fulltext = ''.join(text_list)
        context={
            "status":True,
            "query":slug,
            "url":result_dict,
            "fulltext":fulltext
            
        }
    return context

def duckducksearch(query):
        encoded_query = query.replace(" ","+")
        search_results = DDGS().text(keywords=encoded_query,max_results=4)
        fulltext=""
        text_list = []
        result_url=[]
        for result in search_results:
            print("------------")
            print(result)
            print("")
            text_list.append(result['title'].encode(sys.stdout.encoding, 'replace').decode(sys.stdout.encoding))
            # text_list.append("\n")
            text_list.append(result['body'].encode(sys.stdout.encoding, 'replace').decode(sys.stdout.encoding))
            # text_list.append("\n")
            result_url.append((result['title'].encode(sys.stdout.encoding, 'replace').decode(sys.stdout.encoding),result['href']))
        print(text_list)
        fulltext = ''.join(text_list)
        context={
            "status":True,
            "query":query,
            "url":result_url,
            "fulltext":fulltext
            
        }
        return context

class Duckduckgo(APIView):
    def post(self,request):
        query= request.data['query']
        context= duckducksearch(query)
        return Response(context)


greetingdirectory='assistantapp/models/greeting'

class SemanticSearch(APIView):
    def post(self,request):
        context={}
        query= request.data['query']
        try:
            embeddings.load('assistantapp/models/category')
            res = embeddings.search(query,1)
            print(f"category--- {res}")
            for r in res:

                if r[0]==0:
                    embeddings.load('assistantapp/models/greeting')
                    greetingres = embeddings.search(query,1)
                    print(greetingres)
                    with open("assistantapp/data/greeting.json", "r") as f:
                        greetingdata = json.load(f)["data"]
                    for greeting in greetingres:
                        print(greetingdata[greeting[0]]['answer'])
                        if greeting[1]>.70 :
                            # print(greetingdata[greeting[0]]['question'])
                            context={
                                "status":True,
                                "question":query,
                                "answer":greetingdata[greeting[0]]['answer']
                            }
                        else:
                            context={
                                "status":True,
                                "question":query,
                                "answer":"Hello Sir, How can i help you?"
                            }

                elif r[0]==1 or r[0]==2 or r[0]==3 or r[0]==5:
                        embeddings.load('assistantapp/models/iq')
                        iqres = embeddings.search(query,1)
                        # iqres=''
                        print(iqres)
                        # print(iqres[0][0])
                        
                        with open("assistantapp/data/iq.json", "r") as f:
                            iqdata = json.load(f)
                        if iqres:
                            for iq in iqres:
                                # print(iqdata[iq[0]]['answer'])
                                if iq[1]> .70:
                                    context={
                                            "status":True,
                                            "question":query,
                                            "answer":iqdata[iq[0]]['answer']
                                        }
                                else:
                                    res2= duckducksearch(query)
                                    print(res2)
                                    fulltext=res2['fulltext']
                                    # summary = Summary()
                                    # result = summary(fulltext)
                                    result=T5Summarize(fulltext)
                                    context={
                                            "status":True,
                                            "question":query,
                                            "answer":result
                                        }

                                    print(context)
                                    ndata={
                                            "question":query,
                                            "answer":result
                                        }
                                    
                                    iqdata.append(ndata)

                                    # Write the updated data back to the JSON file
                                    with open('assistantapp/data/iq.json', 'w') as json_file:
                                        json.dump(iqdata, json_file, indent=2)
                                    txtai_data = []
                                    i=0
                                    for text in iqdata:
                                        txtai_data.append((i, text['question'], text['answer']))
                                        i=i+1
                                    print(txtai_data[0])
                                    embeddings.index(txtai_data)
                                    embeddings.save("assistantapp/models/iq")
                                    

                        else:
                        
                            res2= bingsearch(query)
                            print(res2)
                            fulltext=res2['fulltext']
                            # summary = Summary()
                            # result = summary(fulltext)
                            result=T5Summarize(fulltext)
                            context={
                                    "status":True,
                                    "question":query,
                                    "answer":result
                                }

                            print(context)
                            ndata={
                                    "question":query,
                                    "answer":result
                                }
                            
                            iqdata.append(ndata)

                            # Write the updated data back to the JSON file
                            with open('assistantapp/data/iq.json', 'w') as json_file:
                                json.dump(iqdata, json_file, indent=2)
                            txtai_data = []
                            i=0
                            for text in iqdata:
                                txtai_data.append((i, text['question'], text['answer']))
                                i=i+1
                            print(txtai_data[0])
                            embeddings.index(txtai_data)
                            embeddings.save("assistantapp/models/iq")

                # print(f"Text: {data[r[0]]}")
                # print(f"Similarity: {r[1]}")
                # print()
                else:
                    context={
                            "status":False,
                            "question":query,
                            "answer":"Sorry Can't Help with it Right Now"
                        }
        except Exception as e:
            context={
                        "status":False,
                        "question":query,
                        "answer":"Sorry Can't Help with it Right Now"
                    }
        return Response(context)


class CreateModel(APIView):
    def get(self,request):
        try:
            with open("assistantapp/data/greeting.json", "r") as f:
                data = json.load(f)["data"]
            print(len(data))
            txtai_data = []
            i=0
            for text in data:
                txtai_data.append((i, text['question'], text['answer']))
                i=i+1
            print(txtai_data[0])
            embeddings.index(txtai_data)
            embeddings.save("assistantapp/models/greeting")
            context={"Model Created"}
        except Exception as e:
            context={"Model Creation Failed."}
        return Response(context)

class CreateModel2(APIView):
    def get(self,request):
        try:
            with open("assistantapp/data/category.json", "r") as f:
                data = json.load(f)["data"]
            print(len(data))
            txtai_data = []
            i=0
            for text in data:
                txtai_data.append((i, text, None))
                i=i+1
            print(txtai_data[0])
            embeddings.index(txtai_data)
            embeddings.save("assistantapp/models/category")
            context={"Model Created"}
        except Exception as e:
            context={"Model Creation Failed."}
        return Response(context)