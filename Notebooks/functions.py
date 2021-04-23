import pandas as pd
import numpy as np
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
import torch
from torch.utils.data import TensorDataset, random_split

def zero(text):
    l=[0]*len(text)
    return(l)
    

def get_user(l):
    ll=[]
    for x in l:
        ll.append(x['user'])
    return(list(set((ll))))


def index_by_user(l):
    dico = {}
    for user in get_user(l):
        dico[user] = []
        for x in l:
            if x['user'] == user:
                dico[user].append(x['start_offset'])
                dico[user].append(x['end_offset'])
        dico[user].sort()
    return(dico)


def delete_not_read(df):
    data = df.copy()
    columns = list(data.columns)
    columns.remove('index')
    columns.remove('text')
    columns.remove('token')
    l = []
    for user in columns:
        for i in range(len(data)):
                if (data[user][i][1] == 1) and (i not in l):
                    l.append(i)
    
    return(data.drop(l, axis = 0))


def clean_doccano(df):
    data = pd.DataFrame()
    data['text'] = df['text'].apply(lambda x: (" ".join((str(x).split()))).replace('\\',""))
    data['index'] = df['annotations'].apply(index_by_user)
    print("Création de la liste binaire et tokenization...")
    l = []
    data['token'] = ""
    for i in range(len(data)):
        data.loc[i, 'token'] = tokenizer(data['text'][i])['input_ids']
    for x in set(list(df['annotations'].apply(get_user).sum())):
        data[f'{x}'] = data['token'].apply(lambda x: len(x)*[0])
    for i in range(len(data)):
        text=data['text'][i]
        dico = data['index'][i]
        for user in dico.keys():
            index = dico[user]
            if len(index) > 0:
                txt = ""
                txt += text[:index[0]]
                for j in range(len(index)-1):
                    txt += "@"
                    txt +=  text[index[j]:index[j+1]]
                txt += text[index[-1]:]

                token = tokenizer(txt)['input_ids']

                lab = []
                c = 0
                for x in token:
                    if x != 1030:
                        lab.append(c%2)
                    else:
                        c += 1
                data.loc[i, f'{user}'] = lab
            
    print("... suppression des texte non annotés")
    data = delete_not_read(data)
    
    bdd = data.drop(['text', 'index'], axis = 1)
    i=1
    for user in list(bdd.columns)[1:]:
        bdd.rename( columns = {user : f'Annotateur {i}'}, inplace = True)
        i+=1
    return(bdd.reset_index(drop=True))



def split_df(df, cut = 5):
    data = pd.DataFrame()
    for x in df.columns:
        data[x] = ""
        for i in range(len(df)):
            q = len(df[x][i])//cut
            data.loc[i, x] = [df[x][i][:q], df[x][i][q:2*q], df[x][i][2*q:3*q], df[x][i][3*q:4*q], df[x][i][4*q:5*q],df[x][i][5*q:]]
    bd = pd.DataFrame()
    r=0
    for x in df.columns:
        bd[x] = ""
    for i in range(len(data)):
        for x in data.columns:
            a,b,c,d,e,f = data[x][i]
            bd.loc[r, x] = a
            bd.loc[r+1, x] = b
            bd.loc[r+2, x] = c
            bd.loc[r+3, x] = d
            bd.loc[r+4, x] = e
            bd.loc[r+5, x] = f
        r+=6
    for x in bd.columns:
        while min(bd[x].apply(len)) == 0:
            idx = (bd[x].apply(len)).idxmin()
            bd = bd.drop(idx, 0)
    
    bd.reset_index(drop=True,inplace=True)
    bd['attention mask'] = ''
    
    for i in range(len(bd)):        
        n=max(bd['token'].apply(len))
        bd.loc[i, 'attention mask'] = [1] *len(bd['token'][i]) + [0]*(n-len(bd['token'][i]))

    for x in bd.columns:
        n=max(bd[x].apply(len))
        for i in range(len(bd)):
            bd[x][i] += [0]*(n-len(bd[x][i]))

    for x in bd.columns:
        bd[x] = bd[x].apply(lambda x: torch.tensor(x))
        
    return(bd)

def get_train_val(df, cut = 5, train = 0.9):
    
    data = clean_doccano(df)
    data = split_df(data, cut)
    input_ids = torch.stack(list(data['token'].values), 0)
    labels = torch.stack(list(data['Annotateur 1'].values), 0)
    attention_masks = torch.stack(list(data['attention mask'].values), 0)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(train * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return(train_dataset, val_dataset)



    

