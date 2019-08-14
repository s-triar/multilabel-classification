import numpy as np
import pandas as pd
from os import path
import PIL
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import re
import requests
import json


def removeStopwords(teks):
    r = requests.post(
        'http://localhost:9000/stopwords', json={"string": teks})
    data = r.json()['data']
    return data

def formalize(teks):
    r = requests.post(
        'http://localhost:9000/formalizer', json={"string": teks})
    data = r.json()['data']
    return data

def openStopFile(path):
    with open(path, 'r') as f:
        return f.read()

def importData(path, delimeter):
    return pd.read_csv(path, delimiter=delimeter)

def cleaningDate(dataFrame, kol):
    new_kolom = dataFrame[kol].str.split(" ", expand = True) 
    print(new_kolom)
    dataFrame["tgl"]= new_kolom[0] + " "+ new_kolom[1] + " " +new_kolom[2] #+" "+new_kolom[5]
    dataFrame["jam"]= new_kolom[3]  + " " +new_kolom[4]
    dataFrame.drop(columns =[kol], inplace = True) 
    return dataFrame

def drawPlot(title, df, xlabel, ylabel):
    plt.figure(figsize=(15,10))
    df.size().sort_values(ascending=False).plot.bar()
    plt.xticks(rotation=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def drawWordCloud(text):
    wordcloud = WordCloud(max_words=100).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def removeHashTag(doc):
    doc["isi"] = doc.apply(lambda x: re.sub(r'(\#[a-zA-Z0-9]*)', "", x["isi"]), axis=1)
    return doc

def removeNumberAndSymbol(doc):
    doc["isi"] = doc.apply(lambda x: re.sub(r'([\[\]\.\…\?\!\,0-9\/\:\"\(\)]*)', "", x["isi"]), axis=1)
    return doc

def removeUnicode(doc):
    doc["isi"] = doc.apply(lambda x: x["isi"].encode('ascii', errors='ignore').strip().decode('ascii'), axis=1)
    return doc
    

def removeMention(doc):
    doc["isi"] = doc.apply(lambda x: re.sub(r'((RT\s*)*\@[a-zA-Z0-9\_]*)\s*\:*', "", x["isi"]), axis=1)
    return doc

def removeLink(doc):
    doc["isi"] = doc.apply(lambda x: re.sub(r'(http[a-zA-Z0-9\\\-\:\/\.]*)', "", x["isi"]), axis=1)
    return doc

def bundlingTweet(doc):
    paragraph = ""
    for d in doc["isi"]:
        paragraph+=d+"."
    return paragraph