import tkinter as tk
from tkinter import filedialog, Label, Button
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import os

class MLModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Loader")

        # Etichette e pulsanti
        self.model_label = Label(root, text="Nessun modello caricato", fg="red")
        self.model_label.pack(pady=5)

        self.load_model_button = Button(root, text="Carica Modello (.h5)", command=self.load_model)
        self.load_model_button.pack(pady=5)

        self.image_label = Label(root, text="Nessuna immagine selezionata")
        self.image_label.pack(pady=5)

        self.load_image_button = Button(root, text="Carica Immagine", command=self.load_image)
        self.load_image_button.pack(pady=5)

        self.predict_button = Button(root, text="Predici", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(pady=10)

        self.result_label = Label(root, text="", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)

        self.model = None
        self.image_path = None

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("H5 Models", "*.h5")])
        if file_path:
            self.model = load_model(file_path)
            self.model_label.config(text=f"Modello caricato: {file_path.split('/')[-1]}", fg="green")
            self.predict_button.config(state=tk.NORMAL)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleziona un'immagine", 
            initialdir=os.getcwd(),
            filetypes=[("All Files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path).resize((224, 224)).convert('RGB')
            photo = ImageTk.PhotoImage(image)
            
            # Mostra l'immagine nella GUI
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

    def predict(self):
        if self.model and self.image_path:
            image = Image.open(self.image_path).resize((224, 224)).convert('RGB')
            image_array = np.array(image) / 255.0 
            image_array = np.expand_dims(image_array, axis=0)
            
            prediction = self.model.predict(image_array)
            result = f"Predizione: {prediction[0][0]:.4f}"
            self.result_label.config(text=result, fg="blue")

root = tk.Tk()
app = MLModelGUI(root)
root.mainloop()
