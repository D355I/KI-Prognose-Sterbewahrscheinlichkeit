import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model("model.keras")

ls = []


input_data = [0,
              1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

input_data = np.array(input_data).reshape(1, -1)

prediction = model.predict(input_data)
prediction = prediction.item()

def eval_pred(prediction):
    if prediction > 0.8:
        print(f"Hohe wahrscheinlichkeit zu sterben")
        #print(f"Sehr hohe wahrscheinlichkeit zu sterben: {prediction*100:.0f}%")
    elif prediction <= 0.8 and prediction >= 0.6:
        print(f"Bedingte Wahrscheinlichkeit zu sterben") 
    elif prediction < 0.6:
        print(f"Niedrige Wahrscheinlichkeit zu sterben")

eval_pred(prediction)

