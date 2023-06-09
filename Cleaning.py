#!/usr/bin/env python
# coding: utf-8

# In[1]:


# beautiful soup
from bs4 import BeautifulSoup as bs

# Caracter replacement 
import re

# Tokenizer 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words, stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('words')

# Definition des stopwords, après observation des mots à l'aide d'un wordcloud on peut etendre cette liste 
stop_words = stopwords.words('english')
stop_words.extend(['like','want', 'would', 'also', 'however', 'something', 'example', 'one', 'see', 'could', 'trying', 'tried', 'thank', 'thanks'])
stop_words = set(stop_words)


def process_text_vf(doc,min_len_word = 3):
    '''
    
    positionnal arguments : 
    -----------------------
    doc : str : the docuemnt (aka a text in str format) to process
    
    opt args : 
    -----------------------
    rejoin : bool if True return a string else return a list of tokens
    lemm_or_stem : string : if lem do lemmantize else stemmentize
    list_rare_words : list : a list of rare words to exclude
    min_len_word : int : the minimum length of word to not exclude
    
    return :
    -----------------------
    a string (if rejoin is True) or a list of tokens
    '''
        
    #lower
    doc = doc.lower().strip()
    
    # Supprimer le code, flags prend en compte egalement les sauts de ligne 
    doc = re.sub('<code>.*?</code>', '', doc, flags=re.DOTALL)
    
    # Supprimer les balises html de notre corps de question 
    doc = bs(doc, "lxml").text
    
    # Suppresion des retours à la ligne 
    to_clean = re.compile('\n')
    doc = re.sub(to_clean, ' ', doc)

    
    #Supprimer les url
    doc = re.sub(r'http*\S+', '', doc)
    # Supprimer les espace inutiles 
    doc = re.sub('\\s+', ' ', doc)
    # Supprimer les nombres
    doc = re.sub(r'\w*\d+\w*', '', doc)
    
    # tokenize
    tokenizer = RegexpTokenizer(r'\b\w+#?\b')
    raw_tokens_list = tokenizer.tokenize(doc)
    
    
    # classic stop words
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]
    
    
    # no more len words
    more_than_N = [w for w in cleaned_tokens_list if len(w)>= min_len_word]
    
    # lem
    trans = WordNetLemmatizer()
    trans_text = [trans.lemmatize(i) for i in more_than_N]
    
    return trans_text

