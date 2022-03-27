import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup as bs
import requests
from nrclex import NRCLex




st.title('Summary Extraction along with Sentiment Analysis')

Category=  pd.read_csv("Category.csv")

choice = st.sidebar.selectbox('Select Category',Category['Category'])

url2 = 'https://www.goodreads.com/shelf/show/'+str(choice)
req = requests.get(url2)
content = bs(req.content,'html.parser')
book = content.find('div',class_ = 'elementList')
Book_Title = []
for each in book:
    spec = each.find_all_next('a',class_ = 'bookTitle')
    for i in spec:
        Book_Title.append(i.text)
x = np.array(Book_Title[:50])
del Book_Title
Book = st.sidebar.selectbox('Select Book',x)

Book_urls = []
Bookdetails = content.find_all('div', class_ = 'elementList')
for book in Bookdetails:
    book_anchors = book.find('a')
    Book_url = 'https://www.goodreads.com' + book_anchors.get('href')
    Book_urls.append(Book_url)
y = np.array(Book_urls[:50])
del Book_urls


for i,j in enumerate(x):
    if j == str(Book):
        url = y[i]
        st.write(url)
         
        try: 
            req = requests.get(url)
            content = bs(req.content,'html.parser')
            summary = content.find('div',class_ = 'readable stacked')
            summary_text = summary.text
            
            
        
            
            emotion_ = NRCLex(summary_text)
            Emotion =[]
            Score = []
            for key in emotion_.raw_emotion_scores.keys():
                Emotion.append(str(key))
                Score.append(int(emotion_.raw_emotion_scores[key]))
            

            def barplot():
                fig = plt.figure(figsize=(10,4))
                sns.barplot(Emotion,Score) 
                st.pyplot(fig) 
            barplot()  
            del Emotion,Score   
            st.write(f'{emotion_.top_emotions[0][0]} sentiment Emotion represent by summary of book')  

        except:
            st.write('Sorry We are Unable to Find Summary For this Book')     
            
        
     
