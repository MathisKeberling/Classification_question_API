#!/usr/bin/env python
# coding: utf-8

# In[9]:


from flask import Flask
from flask import request
from flasgger import Swagger
from flask_restful import Api, Resource

import joblib

import pandas as pd


# Importer les fonctions de mon fichier de nettoyage
import Cleaning as clean


app = Flask(__name__)
#api = Api(app)

template = {
  "swagger": "2.0",
  "info": {
    "title": "Prédiction de tags sur des questions de StackOverflow",
    "description": "Deployement d'une API qui a pour but de traiter des questions non-traitées, en les nettoyant à l'aide de technique de NLP et en les preparant à l'aide d'un modèle TFIDF. Un regression logistique sera ensuite appliquée.",
    "version": "0.0.1"
  }
}

swagger = Swagger(app, template=template)
# Charger les modèles pré-entrainés
path = "variables/"
multilabel_binarizer = joblib.load(path + "multilabel_binarizer.pkl", 'r')
model = joblib.load(path + "modele_lr_tf_idf.pkl", 'r')
vectorizer = joblib.load(path + "TFIDF_vectorizer.pkl", 'r')

@app.route('/predict', methods=['POST'])

def predict():
    """ 
       Pour utiliser ce code, veuillez simplement copier et coller de façon brute la question dont vous souhaitez obtenir la suggestion de tags
       ---
       parametres:
         - question
               type: string
       retour:
         '200':
           description: Liste de tags prédits et de probabilités associées
           Contenu:
                type: object
                proprietes:
                    Predicted_Tags:
                        type: string
                        description: Liste de tags prédits avec plus de 50% de probabilités (par défaut).
                    Predicted_Tags_Probabilities:
                        type: string
                        description: Liste de tags prédits avec plus de 35% de probabilités
    """
    data = request.get_json()
    question = data['question']
    # Nettoyer la question 
    cleaned_question = clean.process_text_vf(question)
    # Appliquer le transformateur choisi
    X_tfidf = vectorizer.transform([cleaned_question])
    # Prédire les données
    predict = model.predict(X_tfidf)
    predict_probas = model.predict_proba(X_tfidf)
    # Récupérer la target sous forme de string
    tags_predict = multilabel_binarizer.inverse_transform(predict)
    # Dataframe de nos probabilités
    df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
    # Renvoyer la liste des étiquettes d'origine, dans l'ordre correspondant aux colonnes de la représentation binaire
    df_predict_probas['Tags'] = multilabel_binarizer.classes_
    # Affecter valeurs d'un tableau multidimensionnel à une colonne d'un DataFrame
    df_predict_probas['Probas'] = predict_probas.reshape(-1)
    # Selectionner les probabilités  >= 35%
    df_predict_probas = df_predict_probas[df_predict_probas['Probas']>=0.35].sort_values('Probas', ascending=False)
    # Resultats
    results = {}
    results['Predicted_Tags'] = tags_predict
    results['Predicted_Tags_Probabilities'] = df_predict_probas.set_index('Tags')['Probas'].to_dict()    
    return results, 200


#api.add_resource(Autotag, '/autotag/<question>')

if __name__ == "__main__":
    app.run()

