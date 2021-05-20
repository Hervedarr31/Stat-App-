#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import sys
import torch 


# In[ ]:


import functions


# In[ ]:


from torch.utils.data import TensorDataset


# In[4]:


def get_predict_data(df, user, cut=5, train = 1):
    #prend une dataframe (issue de doccano) et la prépare pour trainer.predict()
    data = functions.clean_doccano(df)
    data = functions.split_df(data, cut)
    input_ids = torch.stack(list(data['token'].values), 0)
    attention_masks = torch.stack(list(data['attention mask'].values), 0)
    labels = torch.stack(list(data[f'{user}'].values), 0)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    predict_dataset = dataset
    
    return(predict_dataset)


# In[5]:


def get_decode_data(df,cut=5): #prend en entrée une base de donnée qui ressemble à celle issue de Doccano et lui fait subir les mêmes opérations que pour la prédiction
  data = functions.clean_doccano(df)
  data = functions.split_df(data, cut)
  data['token'] = data['token'].apply(lambda x : x.tolist())
  data['attention mask'] = data['attention mask'].apply(lambda x : x.tolist())

  return(data)


# In[6]:


def decodeur_2(predicted,data_decode): #prend la sortie de trainer.predict et la base originale pour sortir les token correspondants à des labels prédits 1
  a_decoder = []
  for i in range(len(data_decode)):
    for j in range(len(predicted[0])-1):
      if data_decode['attention mask'][i][j] == 1:
        if predicted[i,j]==1:
          a_decoder += [data_decode['token'][i][j]]
        else:
          if predicted[i,j+1]==1:
            a_decoder += [2644] # on ajoute STOP
    #on gère le dernier token
    indice_fin = len(data_decode['token'][0])-1
    if data_decode['attention mask'][i][indice_fin] == 1:
      if predicted[i,indice_fin] ==1:
        a_decoder += [data_decode['token'][i][indice_fin]]
  return(a_decoder)


# def predict_decode_2(df,user):
#   data_predict = get_predict_data(df,user=user) #on prépare les données pour la prédiction
#   prediction = trainer.predict(data_predict)
#   predicted = get_predicted(prediction[0])
#   data_decode = get_decode_data(df) #on prépare la base pour retrouver les token
#   to_decode = decodeur_2(predicted,data_decode)#on récupère les tokens pour les labels 1
# 
#   traduit = tokenizer.decode(to_decode) #on décode les token
# 
#   return(traduit)

# In[ ]:




