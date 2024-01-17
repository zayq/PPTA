import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import tensorflow as tf
import keras
import cv2
import numpy as np

IMG_TAILLE = 50
model_path = os.path.join(os.getcwd(), 'model')
model = keras.models.load_model(model_path)
class_names = ['Positif', 'Negatif']
DOSSIERIMAGES = r'C:\Users\jasmi\Documents\PP\PPTA\produit\C-NMC_Leukemia\training_data\fold_0'
window = tk.Tk()

police = ("Arial", 30, "bold")
prediction_label = tk.Label(window, bg="black")
image_label = tk.Label(window)

def predireImage(ficher):
    new_image = cv2.imread(ficher)
    new_image = cv2.resize(new_image, (IMG_TAILLE, IMG_TAILLE))
    new_image = new_image / 255.0
    new_image = np.expand_dims(new_image, axis=0)

    predictions = model.predict(new_image, verbose=0)
    prediction_class_index = np.argmax(predictions[0])
    prediction_class_nom = class_names[prediction_class_index]
    prediction_probabilite = float(predictions[0][prediction_class_index])
    predicted_probabilite_arrondie = round(prediction_probabilite, 4) * 100
    
    return prediction_class_nom, predicted_probabilite_arrondie
    

def afficherImage(ficher):
    image = Image.open(ficher)

    tk_image = ImageTk.PhotoImage(image)

    image_label.configure(image=tk_image, highlightthickness=0, borderwidth=0)
    image_label.image = tk_image
    image_label.grid(row=0, column=0, pady=20, sticky="n")  
    
    prediction, probabilite = predireImage(ficher)
    print(probabilite)
    prediction_label.config(text=f"Prédiction: {prediction}\n Probabilité: {probabilite}%", font=police, fg="white")
    prediction_label.grid(row=1, column=0, pady=20, sticky="n") 

def handleChargement():    
    ficher = filedialog.askopenfilename(initialdir=DOSSIERIMAGES, filetypes=[("Image files", "*.bmp")])
    if ficher:
        afficherImage(ficher=ficher)

window.title("PP AI LEUKEMIA 2023-2024")
window.geometry("900x800")

upload_button = tk.Button(window, text="Charger une cellule", command=handleChargement, bg="#1c90ed", fg="white", font=("Arial", 22, "bold"), activeforeground="white", activebackground="#1c90ed")
upload_button.grid(row=2, column=0, pady=20, sticky="n")
window.columnconfigure(0, weight=1)
window.configure(bg="black")

window.mainloop()
