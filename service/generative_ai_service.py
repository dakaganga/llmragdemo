import os
import requests
import json
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline
import torch
from vecotrdb import vectordb_con

API_TOKEN = "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" #os.environ["API_TOKEN"] #Set a API_TOKEN environment variable before running
#API_URL = "" #Add a URL for a model of your choosing
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
embedder = pipeline("feature-extraction",model="bert-base-uncased")

def readClaimFiles():
    path_to_json = './data/'
    claimsList=[]
    for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
        with open(path_to_json + file_name) as json_file:
            data = json.load(json_file)
            #print(data)
            claimsList.append(str(data['claimsNarrative']))
    #print(claimsList)
    return claimsList

def generateEmbedings():
    claims=readClaimFiles()
    embedded_docs= [torch.tensor(embedder(doc)[0][0]) for doc in claims]
    return embedded_docs

def findSimilarites(prompt1):
    embedded_prompt=torch.tensor(embedder(prompt1)[0][0]).unsqueeze(0)
    similarites= [float(torch.nn.functional.cosine_similarity(embedded_prompt,d.unsqueeze(0))) for d in generateEmbedings()]
    print(similarites)
    matched_similarities=[]
    for s in similarites:
        if s >0.69:
            matched_similarities.append(similarites.index(s))
    return matched_similarities

def load_data_to_collection():
    collection=vectordb_con.getCollection()
    claims=readClaimFiles()
    claimdocid=1
    for claimtext in claims:
        collection.add(
        documents=[claimtext],
        metadatas=[{"type": "text"}],
        ids=[str(claimdocid)]
        )
        claimdocid = claimdocid+1
    #vectordb_con.add_docs_to_collection(collection) 

def findSimilarites_from_vectordb(prompt1):
   collection=vectordb_con.getCollection()
   #print(prompt1)     
   load_data_to_collection()
   result = vectordb_con.query_collection(collection, prompt1)
   print(result)
   matched_result=result.get("documents")
   
   return matched_result[0]

def buildContext_1(pompt2):
    matched_claim_data=findSimilarites_from_vectordb(pompt2)
    context_2 =""
    for s in matched_claim_data:
        #print(s)
        context_2 += s
    return context_2
'''
def buildContext():
    matched_claim_index=findSimilarites(question)
    context_2 =""
    for s in matched_claim_index:
        context_2 += readClaimFiles()[s]
    return context_2
'''
def query(question):
    context_1=buildContext_1(question)
    prompt = f"""Use the following context to answer the question at the end. No other questions and answers include in the response. answer should be short

    {context_1}

    Question: {question}
    """

    payload = {
        "inputs": prompt,
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.9,
            "top_p": 0.8,
            "do_sample": False,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']

#question ='list oout the patient names who had comonoscopy, what procedure performed to jane doe'
#question ='list out the patient names who had colonoscopy' who had the nasal allergy
#print(query(question))
