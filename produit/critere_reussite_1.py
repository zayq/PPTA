import keras
import cv2
import numpy as np
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

IMG_TAILLE = 50
DOSSIERIMAGES = r'C:\Users\jasmi\Documents\PP\PPTA\produit\C-NMC_Leukemia\validation_data\fold_0'
CATEGORIES = ['Positif', 'Negatif']

model_path = os.path.join(os.getcwd(), 'model')
model = keras.models.load_model(model_path)

bonresultat = 0
mauvaisresultat = 0

results_data = []

for categorie in CATEGORIES: 
    ficher = os.path.join(DOSSIERIMAGES, categorie)
    label = CATEGORIES.index(categorie) 
    for img in os.listdir(ficher): 
        try: 
            img_path = os.path.join(ficher, img) 
            n_image = cv2.imread(img_path)
            n_image = cv2.resize(n_image, (IMG_TAILLE, IMG_TAILLE))
            n_image = n_image / 255.0
            n_image = np.expand_dims(n_image, axis=0)

            predictions = model.predict(n_image, verbose=0)

            class_names = ['Positif', 'Negatif']
            prediction_class_index = np.argmax(predictions[0])
            prediction_class = class_names[prediction_class_index]
            
            if prediction_class == categorie:
                bonresultat += 1
            else:
                mauvaisresultat += 1
            
            results_data.append({
                'Image': img,
                'Catégorie': categorie,
                'Prédiction': prediction_class,
                'État de la Prédiction': prediction_class == categorie
            })
        except Exception as e: 
            print('Error:', str(e)) 

results_df = pd.DataFrame(results_data)

excel_path = os.path.join(os.getcwd(), 'results.xlsx')
results_df.to_excel(excel_path, index=False)

print("Bon resultats: " + str(bonresultat))
print("Mauvais resultats: " + str(mauvaisresultat))
print("Total results: " + str(mauvaisresultat + bonresultat))
print(f"Results saved to {excel_path}")
