import numpy as np
import pickle

loaded_model = pickle.load(open('rainforest_model.sav', 'rb'))

# Make predictions
input_data = [-0.47415027,0.6625177,0.41563237,-0.46888417,0.7290459,0.59471226,0.5436662]
input_data = np.array(input_data).reshape(1, -1)
prediction = loaded_model.predict(input_data)
print(prediction)