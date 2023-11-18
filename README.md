# Illness-Classifier

### Overview
The classifier is trained on a dataset ('symptoms.csv') containing symptom information for different types of illnesses: ALLERGY, COLD, COVID, and FLU. The neural network model is trained to predict the illness type based on the symptoms provided.

### Requirements
Libraries and Tools
Pandas
NumPy
Keras
Joblib
Scikit-learn
Dataset
The dataset used for training the model is 'symptoms.csv'.

### Usage
Training the Model
  1. Run the provided Python script (symptoms_classifier.py) to train the neural network model using the dataset.
  2. The model will be saved as 'Symptoms_Keras_Model.h5'.

Making Predictions
  1. Loading the model using keras.models.load_model("Symptoms_Keras_Model.h5").
  2. Predicting the illness type using the trained model.

User Input for Prediction
  1. The project provides a way for users to input their symptoms for prediction.
  2. Run the provided Python script (user_input_predict.py) to interactively input symptoms and get a prediction.
Files Included
  - symptoms.csv: Dataset containing symptom information.
  - Symptoms_Keras_Model.h5: Trained neural network model.
  - symptoms_classifier.py: Script for training the model.
  - user_input_predict.py: Script for allowing user input for predictions.
  - 
How to Use
  1. Clone the repository: git clone <repository_URL>
  2. Install the necessary dependencies (pip install -r requirements.txt).
  3. Train the model or use the provided trained model for predictions.
  4. Follow the instructions in user_input_predict.py to input symptoms and get predictions interactively.

### Acknowledgments
The project utilizes Keras and Scikit-learn for model creation and evaluation.
The dataset used is sourced from kaggle
