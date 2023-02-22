import numpy as np
import pandas as pd

def read_data():
    data = pd.read_csv('core/data/Dt_Hasil_Preprocessing.csv')
    data = data[['tweet', 'label']]
    return data

def read_textblob():
    data2 = pd.read_csv('core/data/Label_Textblob.csv')
    data2 = data2[['tweet', 'TextBlob']]
    return data2

def read_vader():
    data3 = pd.read_csv('core/data/Label_Vader.csv')
    data3 = data3[['tweet','Sentiments']]
    return data3

def read_cleansing():
    data4 = pd.read_csv('core/data/Dt_Hasil_Preprocessing.csv')
    data4 = data4[['tweet', 'Text_cleansing']]
    return data4

def read_spelling():
    data45 = pd.read_csv('core/data/Dt_Hasil_Preprocessing.csv')
    data45 = data45[['Text_cleansing', 'Text_slang']]
    return data45

def read_stopword():
    data5 = pd.read_csv('core/data/Dt_Hasil_Preprocessing.csv')
    data5 = data5[['Text_cleansing', 'Text_stopword']]
    return data5

def read_stemming():
    data6 = pd.read_csv('core/data/Dt_Hasil_Preprocessing.csv')
    data6 = data6[['Text_stopword', 'Text_stemming']]
    return data6

def read_tokenizing():
    data7 = pd.read_csv('core/data/Dt_Hasil_Preprocessing.csv')
    data7 = data7[['Text_stemming', 'tokens']]
    return data7

if __name__=="__main__":
    read_data()