from tracemalloc import stop
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup as bs
import requests

st.title('Summary Extraction along with Sentiment Analysis')

Category=  pd.read_csv("Category.csv")
# select Category of Book like Sport,Art,Science etc.
choice = st.sidebar.selectbox('Select Category',Category['Category'])
#url1 = 'https://www.goodreads.com/shelf/show/'+str(choice)
def get_choice(url1):
    
    #selecting Book Title
    try:
        Book_Title = []
        req = requests.get(url1)
        content = bs(req.content,'html.parser')
        book = content.find('div',class_ = 'elementList')
        for each in book:
            spec = each.find_all_next('a',class_ = 'bookTitle')
            for i in spec:
                Book_Title.append(i.text)
        x = Book_Title[:50]
        del Book_Title
        
        return x
        

    except:
        return st.write("Please Check Your Internet Connection")
        
        
Book_Title = get_choice('https://www.goodreads.com/shelf/show/'+str(choice))


if Book_Title != None:
    Book = st.sidebar.selectbox('Select Book',Book_Title)
    #get Book Summary:
    Book_urls = []
    req = requests.get('https://www.goodreads.com/shelf/show/'+str(choice))
    content = bs(req.content,'html.parser')
    
    Bookdetails = content.find_all('div', class_ = 'elementList')
    for book in Bookdetails:
        book_anchors = book.find('a')
        Book_url = 'https://www.goodreads.com' + book_anchors.get('href')
        Book_urls.append(Book_url)
    y = np.array(Book_urls[:50])
    del Book_urls


    for i,j in enumerate(Book_Title):
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



    
    # for i,j in enumerate(df['Book_Title']):
    #     if j==str(book):
    #         url = df['Book_Url'][i]
    #         req = requests.get(url)
    #         content = bs(req.content,'html.parser')  
    #         summary = content.find('div',class_ = 'readable stacked')
    #         st.write(summary.text)



# else:
#     stop

    

