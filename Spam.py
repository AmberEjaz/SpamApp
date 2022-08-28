# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 09:12:01 2022

@author: HP
"""


import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import sys
pd.options.mode.chained_assignment = None
from nltk.corpus import stopwords
import warnings

uploaded_file = st.file_uploader("SMS_data.csv")
if uploaded_file is not None:
     bytes_data = uploaded_file.getvalue()
     st.write(bytes_data)
     dataframe = pd.read_csv(uploaded_file)
     st.write(dataframe)

df = pd. read_csv ('SMS_data.csv',encoding= 'unicode_escape')
df["text_lower"] = df["Message_body"].str.lower()
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["text_wo_punct"] = df["text_lower"].apply(lambda text: remove_punctuation(text))
", ".join(stopwords.words('english'))

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["text_wo_stop"] = df["text_wo_punct"].apply(lambda text: remove_stopwords(text))
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df["text_stemmed"] = df["text_wo_stop"].apply(lambda text: stem_words(text))

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
df["text_lemmatized"] = df["text_stemmed"].apply(lambda text: lemmatize_words(text))


#cnt = Counter()
#for text in df["text_wo_stop"].values:
    #for word in text.split():
        #cnt[word] += 1

cnt = Counter()
for text in df[df['Label']=='Spam']['text_lemmatized'].values:
    for word in text.split():
        cnt[word] += 1
        
s=cnt.most_common(10)


cnt2 = Counter()
for text in df[df['Label']=='Non-Spam']['text_lemmatized'].values:
    for word in text.split():
        cnt2[word] += 1
        
ns=cnt2.most_common(10)

      
#cnt.most_common(10)
#print(cnt)

def main():
    st.title('Classification of Texts')
    data=st.selectbox('Select Type Of Messages', df['Label'].unique())
    b=st.button('show data')
    if b:
        col1, col2=st.columns(2)
        if data == 'Spam':
            with col1:
                words = [word for word, _ in s]
                counts = [counts for _, counts in s]
                plt.barh(words, counts)
                plt.title("Most frequent words in Spam Messages")
                plt.ylabel("Frequency")
                plt.xlabel("Words")
                st.pyplot(plt)
                #st.bar_chart(x=words, y=counts, width=0, height=0, use_container_width=True)
        else:
            with col1:
                words = [word for word, _ in ns]
                counts = [counts for _, counts in ns]
                plt.barh(words, counts)
                plt.title("Most frequent words in Non Spam Messages")
                plt.ylabel("Frequency")
                plt.xlabel("Words")
                st.pyplot(plt)
                #st.bar_chart(data=chart_data,x=None, y=None, width=0, height=0, use_container_width=True)
            
        with col2:
            num=df.groupby('Date_Received')['Message_body'].count()
            new=pd.DataFrame(num)
            less=new.sort_values('Date_Received', ascending=False).head(15)
            fig = plt.figure(figsize=(20, 4))
            sns.lineplot(data=less, x="Date_Received", y="Message_body")   
            st.pyplot(fig)
                
        #col1, col2=st.columns(2)
        #st.table(data)        
        #with col1:
           #graph=cnt.most_common(10)
           
           #st.bar_chart(data=graph, x=None, y=None, width=0, height=0, use_container_width=True)
        #with col2:
            #chart=
            #st.line_chart(data=chart, *, x=None, y=None, width=0, height=0, use_container_width=True)
if __name__ == '__main__':
    main()