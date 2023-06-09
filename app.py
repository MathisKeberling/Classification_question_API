#!/usr/bin/env python
# coding: utf-8

# In[9]:


from flask import Flask
from flasgger import Swagger
from flask_restful import Api, Resource

import joblib

import pandas as pd

# USE
import tensorflow_hub as hub 

# Importer les fonctions de mon fichier de nettoyage
import Cleaning as clean


app = Flask(__name__)
api = Api(app)

template = {
  "swagger": "2.0",
  "info": {
    "title": "Prédiction de tags sur des questions de StackOverflow",
    "description": "Deployement d'une API qui a pour but de traiter des questions non-traitées, en les nettoyant à l'aide de technique de NLP et en les preparant à l'aide d'un modèle USE. Un regression logistique sera ensuite appliquée.",
    "version": "0.0.1"
  }
}

swagger = Swagger(app, template=template)
# Charger les modèles pré-entrainés
path = "variables/"
multilabel_binarizer = joblib.load(path + "multilabel_binarizer.pkl", 'r')
model = joblib.load(path + "lr_use.pkl", 'r')

# charger le modèle Universal Sentence Encoder
module_url = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class Autotag(Resource):
    def get(self, question):
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
        # Nettoyer la question 
        cleaned_question = clean.process_text_vf(question)
        
        # Transformer notre question
        cleaned_question = ' '.join(cleaned_question)
        cleaned_question = [cleaned_question]
        
        # Appliquer le transformateur choisi
        X_use = module_url(cleaned_question)
        
        
        # Prrédire les données
        predict = model.predict(X_use)
        predict_probas = model.predict_proba(X_use)
        
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


api.add_resource(Autotag, '/autotag/<predictions>')

if __name__ == "__main__":
    app.run()

