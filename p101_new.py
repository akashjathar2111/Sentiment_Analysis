import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup as bs
import requests


st.title('Summary Extraction along with Sentiment Analysis')

Category=  pd.read_csv("Category.csv")

choice = st.sidebar.selectbox('Select Category',Category['Category'])

url2 = 'https://www.goodreads.com/shelf/show/'+str(choice)
req = requests.get(url2)
content = bs(req.content,'html.parser')
Book_Title = []
book = content.find('div',class_ = 'elementList')
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
        req = requests.get(url)
        content = bs(req.content,'html.parser') 
        try: 
            summary = content.find('div',class_ = 'readable stacked')
            st.write(summary.text)
        except:
            st.write('Sorry We are Unable to Find Summary For this Book') 

        


    

