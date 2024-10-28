import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('rainforest_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

# Define the mapping from binary output to crop names
crop_mapping = {
    (0, 0, 0, 0, 1): 'orange',
    (0, 0, 0, 1, 0): 'grapes',
    (0, 0, 1, 1, 0): 'kidneybeans',
    (0, 1, 0, 0, 0): 'mungbean',
    (0, 1, 0, 0, 1): 'coffee',
    (0, 1, 0, 1, 1): 'apple',
    (0, 1, 1, 0, 0): 'blackgram',
    (0, 1, 1, 1, 0): 'maize',
    (0, 1, 1, 1, 1): 'rice',
    (1, 0, 0, 0, 0): 'watermelon',
    (1, 0, 0, 1, 0): 'mango',
    (1, 0, 0, 1, 1): 'pomegranate',
    (1, 0, 1, 0, 0): 'papaya',
    (1, 0, 1, 0, 1): 'coconut',
    (1, 0, 1, 1, 0): 'chickpea',
    (1, 0, 0, 0, 1): 'jute',
    (1, 0, 0, 0, 0): 'watermelon',
    (1, 1, 0, 0, 0): 'jute',
    # Add more mappings based on your data if necessary
}

def get_crop_name(prediction):
    # Convert the prediction to a tuple
    binary_output = tuple(prediction[0])
    # Get the crop name from the mapping
    return crop_mapping.get(binary_output, "Unknown crop")

def prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = loaded_scaler.transform(input_data)  # Scale the input data
    prediction = loaded_model.predict(input_data_scaled)
    return prediction


def main():
    #title for web page
    st.title('Crop Recommendation System')

    #Getting the input data from the user
    N = st.text_input('N')
    P = st.text_input('P')
    K = st.text_input('K')
    temperature = st.text_input('Temperature')
    humidity = st.text_input('Humidity')
    ph = st.text_input('ph')
    rainfall = st.text_input('rainfall')

    # Initialize recommended_crop variable
    recommended_crop = None
    
    # creating a button for Prediction
    
    if st.button('Recommend Crop'):
        prediction_result = prediction([N,P,K,temperature,humidity,ph,rainfall])
        recommended_crop = get_crop_name(prediction_result)
        
        
    # Display the recommendation only if it's not None
    if recommended_crop is not None:
        st.success(f'Recommended Crop: {recommended_crop}')
    
    
    
    
    
if __name__ == '__main__':
    main()
    