import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox

data = pd.read_csv('medical_insurance.csv')
data = data.drop(columns = ['region'])
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

X = data.drop('charges', axis=1).values
y = data['charges'].values

np.random.seed(0)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = X[train_indices], y[train_indices]

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def predict(X_train, y_train, x_test):
    distances = np.array([euclidean_distance(x_test, x_train) for x_train in X_train])
    min_index = np.argmin(distances)
    return y_train[min_index]

def make_prediction():
    try:
        inputs = [
            float(entry_age.get()),
            1 if var_sex.get() == 'male' else 0,
            float(entry_bmi.get()),
            int(entry_children.get()),
            1 if var_smoker.get() == 'yes' else 0
        ]
        
        x_test = np.array(inputs)
        result = predict(X_train, y_train, x_test)
        
        messagebox.showinfo("Resultado de la Predicción", f"El costo estimado es: ${result:.2f}")

    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores válidos en todos los campos.")

root = tk.Tk()
root.title("Predicción de Cargos Médicos")

tk.Label(root, text="Edad:").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Sexo:").grid(row=1, column=0)
var_sex = tk.StringVar(value='male')
tk.OptionMenu(root, var_sex, 'male', 'female').grid(row=1, column=1)

tk.Label(root, text="BMI:").grid(row=2, column=0)
entry_bmi = tk.Entry(root)
entry_bmi.grid(row=2, column=1)

tk.Label(root, text="Número de Hijos:").grid(row=3, column=0)
entry_children = tk.Entry(root)
entry_children.grid(row=3, column=1)

tk.Label(root, text="Fumador:").grid(row=4, column=0)
var_smoker = tk.StringVar(value='no')
tk.OptionMenu(root, var_smoker, 'yes', 'no').grid(row=4, column=1)

predict_button = tk.Button(root, text="Predecir Cargos", command=make_prediction)
predict_button.grid(row=5, column=0, columnspan=2)

root.mainloop()
