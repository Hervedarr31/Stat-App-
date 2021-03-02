
Réu du 17/02/21

Compte Doccano
http://doccanno.huma-num.fr/
Username : editorialistes1
Password : statapp2021

Username: editorialistes2
Password: statapp2021

GIthub : language-power

Core/TransferSociologist/data.py
Core/TransferSociologist/models.py


Pour du sequence labelling on démarre avec du LSTM
Avec KERAS 

Pour échiquier pol: 
Doc 2 vec ou 
PCA2 


Doccanon 

Modèles basés sur transformers pré-entrainés =modèle de langage 

Différents types de fine-tuning:

 Sequence classification  (Analyse de sentiments = cas particulier )
Sequence labelling (off ou pas off, reconnaissance d’entités
Sequence to sequence (possible with different size in output )




Chose abstraite besoin d’une centaine (ou 200) annotations
Moins abstrait (50 ou 100) 

Hugging face.co/Transformers
Transformers = bibliothèque avec plein de modèles que l’on peut download en une fois 
Modèle de langue français pour Analayse sentiment (Flaubert ou Camembert)

Utiliser pytorch plutôt que tensor flow pour débuter plus facile 

Doccano pour annoter les textes 

Package pigeon pour les classifications


Réu Précédente : 
Présentation du "nouveau" NLP 

Critères pour Edito 
https://compol.cnrs.fr/?p=66

Re-Bonsoir à vous quatre, 

- Comme promis, les slides que je vous ai montrées en pj. N'hésitez pas si vous avez des questions dessus.

 - Côté "self supervised learning", si vous voulez un peu vous amuser avec du word2vec sans avoir à tout recoder, et voir le genre de pertinence linguistique d'un objectif d'apprentissage self-supervised simple, le package gensim contient une classe "word2vec" facile à utiliser : 
https://radimrehurek.com/gensim/models/word2vec.html 

- Coté fine-tuning de modèles type BERT : voilà la page de camemBERT 
https://huggingface.co/transformers/model_doc/camembert.html   

- Un tuto pour fine-tuner BERT dans la pratique (pour passer à camemBERT je vous laisse adapter)
https://mccormickml.com/2019/07/22/BERT-fine-tuning/  Peut-être que c'est mieux que je vous laisse travailler à partir de ce tuto pour que vous maîtrisiez mieux le code step-by-step plutôt que vous envoyer déjà mon code  "tout prêt". N'hésitez pas si vous avez des questions


 - Pour l'annotation en classif très lightweight (mais instable, pas multiannotateur etc) sur un jupyter notebook :
https://pypi.org/project/pigeon-jupyter/

- Pour Doccanno, il faudra voir avec Etienne si on peut vous mettre  sur notre instance humanum ou si on ouvre une nouvelle instance.


En vous souhaitant une bonne soirée, et à dans 2 semaines, 
