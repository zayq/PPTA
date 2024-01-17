import numpy as np # Numpy est la librarie principal pour utiliser les matrices additionner multiplier ect.
import cv2 # cv2 pour transformer mes images en tant que matrices
import os # os pour avoir acces a "loperating system" sert par exemple a recuperer des fichier
import random # je vais lutiliser pour melanger mes images
import matplotlib.pyplot as plt # pour visualiser les images
import pickle # pour stocker nos matrices en tant que ficher

DOSSIERIMAGES = r'C:\Users\jasmi\Documents\PP\PPTA\produit\C-NMC_Leukemia\training_data\fold_0' # Le folder avec les images de trainnig

CATEGORIES = ['Positif', 'Negatif']

IMG_TAILLE = 50


data = []

for category in CATEGORIES: 
    folder = os.path.join(DOSSIERIMAGES, category)
    label = CATEGORIES.index(category) 
    for img in os.listdir(folder): 
        try: 
            img_path = os.path.join(folder, img) 
            img_en_liste = cv2.imread(img_path) 
            img_en_liste = cv2.resize(img_en_liste, (IMG_TAILLE, IMG_TAILLE)) 
            data.append([img_en_liste, label]) 
        except Exception as e: 
            print('Error:', str(e)) 
        
   
   
random.shuffle(data) 

x = []
y = []

for nom, label in data:
    x.append(nom)
    y.append(label)

x = np.array(x)
y = np.array(y)

pickle.dump(x, open('x.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))