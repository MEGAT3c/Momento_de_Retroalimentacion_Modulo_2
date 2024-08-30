import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Cargar los datos (ajusta la ruta del archivo)
df = pd.read_csv('gender_classification_v7.csv')  
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Dividir los datos en características (X) y etiqueta (y)
X = df.drop('gender', axis=1).values
y = df['gender'].values

# Dividir en conjunto de entrenamiento y prueba
np.random.seed(0)  
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = X[train_indices], y[train_indices]

# Función para calcular la distancia euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Clasificador k-NN con k=1
def predict(X_train, y_train, x_test):
    distances = np.array([euclidean_distance(x_test, x_train) for x_train in X_train])
    min_index = np.argmin(distances)
    return y_train[min_index]

# Función para manejar la predicción desde la interfaz gráfica
def make_prediction():
    try:
        # Obtener los valores de las entradas
        inputs = [
            int(entry_long_hair.get()),
            float(entry_forehead_width.get()),
            float(entry_forehead_height.get()),
            int(entry_nose_wide.get()),
            int(entry_nose_long.get()),
            int(entry_lips_thin.get()),
            int(entry_distance_nose_to_lip.get())
        ]
        
        # Convertir a un array numpy
        x_test = np.array(inputs)
        
        # Realizar la predicción
        result = predict(X_train, y_train, x_test)
        
        # Mostrar resultado
        gender = 'Femenino' if result == 1 else 'Masculino'
        messagebox.showinfo("Resultado de la Predicción", f"El género estimado es: {gender}")

    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores válidos en todos los campos.")

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Predicción de Género")

# Crear etiquetas y campos de entrada para cada característica
tk.Label(root, text="Longitud del cabello (0/1):").grid(row=0, column=0)
entry_long_hair = tk.Entry(root)
entry_long_hair.grid(row=0, column=1)

tk.Label(root, text="Ancho de la frente (cm):").grid(row=1, column=0)
entry_forehead_width = tk.Entry(root)
entry_forehead_width.grid(row=1, column=1)

tk.Label(root, text="Altura de la frente (cm):").grid(row=2, column=0)
entry_forehead_height = tk.Entry(root)
entry_forehead_height.grid(row=2, column=1)

tk.Label(root, text="Nariz ancha (0/1):").grid(row=3, column=0)
entry_nose_wide = tk.Entry(root)
entry_nose_wide.grid(row=3, column=1)

tk.Label(root, text="Nariz larga (0/1):").grid(row=4, column=0)
entry_nose_long = tk.Entry(root)
entry_nose_long.grid(row=4, column=1)

tk.Label(root, text="Labios finos (0/1):").grid(row=5, column=0)
entry_lips_thin = tk.Entry(root)
entry_lips_thin.grid(row=5, column=1)

tk.Label(root, text="Distancia nariz a labios (0/1):").grid(row=6, column=0)
entry_distance_nose_to_lip = tk.Entry(root)
entry_distance_nose_to_lip.grid(row=6, column=1)

# Botón para realizar la predicción
predict_button = tk.Button(root, text="Predecir Género", command=make_prediction)
predict_button.grid(row=7, column=0, columnspan=2)

# Iniciar la interfaz gráfica
root.mainloop()
