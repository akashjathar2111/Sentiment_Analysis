import pandas as pd
import streamlit as st
import bs4
from bs4 import BeautifulSoup as bs
import requests


st.title('Summary Extraction along with Sentiment Analysis')

Category=  pd.read_csv(r"C:\Users\Shri Ganesha\Documents\Excelr_data\project-101\Category.csv")

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
x = Book_Title
del Book_Title
Book = st.sidebar.selectbox('Select Book',x)

Book_urls = []
Bookdetails = content.find_all('div', class_ = 'elementList')
for book in Bookdetails:
    book_anchors = book.find('a')
    Book_url = 'https://www.goodreads.com' + book_anchors.get('href')
    Book_urls.append(Book_url)
y = Book_urls
del Book_urls

df = pd.DataFrame()
df['Book_Title'] = x[:50]
df['Book_urls'] = y[:50]

for i,j in enumerate(df):
    if Book == df['Book_Title'][i]:
        for i,j in enumerate(df['Book_Title']):
            if j == str(Book):
                url = df['Book_urls'][i]
                req = requests.get(url)
                content = bs(req.content,'html.parser')  
                summary = content.find('div',class_ = 'readable stacked')
                st.write(summary.text)
