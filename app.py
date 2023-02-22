from os import link
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import contractions
from sklearn.pipeline import Pipeline
import joblib
import string
import pickle
import re
import sys
import json
import base64

nltk.download('stopwords')
from nltk.corpus import stopwords
list_stopwords = set(stopwords.words('indonesian'))
from nltk import word_tokenize
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk import word_tokenize
import emoji 

from core.read_data import read_data, read_spelling, read_textblob, read_vader, read_cleansing, read_stopword, read_tokenizing, read_stemming

app = Flask(__name__)
app.config['SECRET_KEY']='main'
socket=SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/label_1')
def label_1():
    data = read_data()
    return render_template('label_1.html', nama_kolom=data.columns.values, datas=list(data.values.tolist()), kolom_index='Tweet', zip=zip)

@app.route('/label_2')
def label_2():
    data2 = read_textblob()
    return render_template('label_2.html', nama_kolom=data2.columns.values, datas=list(data2.values.tolist()), kolom_index='tweet', zip=zip)

@app.route('/label_3')
def label_3():
    data3 = read_vader()
    return render_template('label_3.html', nama_kolom=data3.columns.values, datas=list(data3.values.tolist()), kolom_index='tweet', zip=zip)

@app.route('/cleansing')
def cleansing():
    data4 = read_cleansing()
    return render_template('cleansing.html', nama_kolom=data4.columns.values, datas=list(data4.values.tolist()), kolom_index='tweet', zip=zip)

@app.route('/spelling')
def spelling():
    data45 = read_spelling()
    return render_template('spelling.html', nama_kolom=data45.columns.values, datas=list(data45.values.tolist()), kolom_index='tweet', zip=zip)


@app.route('/stopword')
def stopword():
    data5 = read_stopword()
    return render_template('stopword.html', nama_kolom=data5.columns.values, datas=list(data5.values.tolist()), kolom_index='tweet', zip=zip)

@app.route('/stemming')
def stemming():
    data6 = read_stemming()
    return render_template('stemming.html', nama_kolom=data6.columns.values, datas=list(data6.values.tolist()), kolom_index='tweet', zip=zip)

@app.route('/tokenizing')
def tokenizing():
    data7 = read_tokenizing()
    return render_template('tokenizing.html', nama_kolom=data7.columns.values, datas=list(data7.values.tolist()), kolom_index='tweet', zip=zip)


@app.route('/pengujian')
def pengujian():
    return render_template('pengujian.html')


# Load trained Pipeline
load_model = joblib.load('model_svm.pkl')
load_model2 = joblib.load('model_svm_tb.pkl')
load_model3 = joblib.load('model_vader_svm.pkl')

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')


@app.route('/hasilnya',methods=["GET"])
def hasilnya():
    subject = request.args.get("sub")
    subject = [subject]
    print(subject)
    result = {}
    

    def cleansing(tokens):
        #Hapus @username
        t1 = re.sub('\B@\w+', "", tokens)
        #Remove emoji 
        t2 = re.sub(r'[^\x00-\x7f]', r'', t1) #Remove non ASCII chars
        t22 = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', t2)
        #Hapus url (http:// atau https://)
        t3 = re.sub('(http|https):\/\/\S+', '', t22)
        #Replace hashtag
        t4 = re.sub('#+', '', t3)
        #Lower case setiap kata
        t5 = t4.lower()
        #Hapus kata berulang (ooooo jadi o)
        t6 = re.sub(r'(.)\1+', r'\1\1', t5)
        #Hapus punctuation spt !!!!!!!! jadi !
        t7 = re.sub('r[\?\.\!]+(?=[\?.\!])', '', t6)
        #Hapus nomor & karakter spesial
        t8 = re.sub(r'[^a-zA-Z]', ' ', t7)
        #Hapus singkatan/contractions dengan yg panjang
        t9 = contractions.fix(t8)
        return t9

    test_cleansing=[]
    for i in range(0, len(subject)):
        test_cleansing.append(cleansing(subject[i]))

    result ['cleansing'] = ' '.join(list(map(lambda x: str(x), test_cleansing)))
    cleansing = result ['cleansing']

    def open_kamus_prepro(x):
        kamus={}
        with open(x,'r') as file :
            for line in file :
                slang=line.replace("'","").split(':')
                kamus[slang[0].strip()]=slang[1].rstrip('\n').lstrip()
        return kamus

    kamus_slang = open_kamus_prepro('kamus_spelling_word.txt')

    def slangword(text):
        sentence_list = text.split()
        new_sentence = []
        
        for word in sentence_list:
            for candidate_replacement in kamus_slang:
                if candidate_replacement == word:
                    word = word.replace(candidate_replacement, kamus_slang[candidate_replacement])
            new_sentence.append(word)
        return " ".join(new_sentence)

    test_slangword=[]
    for i in range(0,len(test_cleansing)):
        test_slangword.append(slangword(test_cleansing[i]))

    result ['slangword'] = ' '.join(list(map(lambda x: str(x), test_slangword)))
    slangword = result ['slangword'] 
    
    def remove_stopword(slangword):
        slangword = ' '.join(['' if word in list_stopwords else word for word in slangword.split(' ')])
        slangword = re.sub(' +', ' ', slangword) #remove extra spaces
        slangword = slangword.strip()
        return slangword
    
    test_stopword=[]
    for i in range(0, len(test_slangword)):
        test_stopword.append(remove_stopword(test_slangword[i]))

    result ['stopword'] = ' '.join(list(map(lambda x: str(x), test_stopword)))
    stopword = result ['stopword']


    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def text_stemming(stopword):  
        return stemmer.stem(stopword)
 
    test_stemming=[]
    for i in range(0, len(test_stopword)):
        test_stemming.append(text_stemming(test_stopword[i]))

    result ['stemming'] = ' '.join(list(map(lambda x: str(x), test_stemming)))
    stemming = result ['stemming']

    result['tokenize'] = [word_tokenize(sen) for sen in test_stemming]
    tokenize = result['tokenize']

    result['text_final'] = [' '.join(sen) for sen in tokenize]
    text_final_ = result['text_final']
    
    
    predictions = load_model.predict(text_final_)[0]

    predictions2 = load_model2.predict(text_final_)[0]
    predictions3 = load_model3.predict(text_final_)[0]
    

    result['predict'] = 'Negatif' if predictions == -1 else 'Positif' if predictions == 1 else 'Netral'
    hasil_kelas = result['predict']

    result['predict2'] = 'Negatif' if predictions2 == -1 else 'Positif' if predictions2 == 1 else 'Netral'
    hasil_kelas2 = result['predict2']

    result['predict3'] = 'Negatif' if predictions3 == -1 else 'Positif' if predictions3 == 1 else 'Netral'
    hasil_kelas3 = result['predict3']
    

    return render_template("hasilnya.html", 
                                subject = subject,
                                cleansing = cleansing,
                                slangword=slangword,
                                stopword = stopword,
                                stemming = stemming,
                                tokenize = tokenize,
                                text_final_ = text_final_,
                                hasil_kelas = hasil_kelas,
                                hasil_kelas2 = hasil_kelas2,
                                hasil_kelas3 = hasil_kelas3 
                                ) 

    
if __name__=="__main__":
    socket.run(app)