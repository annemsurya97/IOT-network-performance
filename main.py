from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, joblib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

global filename
global classifier
global X, y, X_train, X_test, y_train, y_test ,Predictions
global dataset, df1, df2, sc, train_or_load_dnn, dnn_model
global le, labels

def upload():
    global filename
    global dataset, df1
    filename = filedialog.askopenfilename(initialdir = "Datasets")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    df1= pd.read_csv(filename)
    text.insert(END,'\n\n IoT Network Dataset: \n', str(df1))
    text.insert(END,df1)

def preprocess():
    global dataset, df1, df2
    global X, y, X_train, X_test, y_train, y_test, sc, le, labels
    text.delete('1.0', END)

    # Display basic information about the dataset
    #text.insert(END, '\n\nInformation of the dataset: \n', str(df1.info()))
    print(df1.info())
    text.insert(END, '\n\nDescription of the dataset: \n' + str(df1.describe().T))
    text.insert(END, '\n\nChecking null values in the dataset: \n' + str(df1.isnull().sum()))
    text.insert(END, '\n\nUnique values in the dataset: \n' + str(df1.nunique()))
    
        
    # Function to fill NaNs with the most frequent value
    def fill_with_mode(df):
        for column in df.columns:
            most_frequent_value = df1[column].mode()[0]
            df1[column].fillna(most_frequent_value, inplace=True)
    
    # Apply the function
    fill_with_mode(df1)
    
    
    labels=[
    "normal",            # 0
    "wrong setup",       # 1
    "ddos",              # 2
    "Data type probing", # 3
    "scan attack",       # 4
    "man in the middle"  # 5
    ]

  
    y = df1['normality'].values
    X = df1.drop(columns=['normality'], axis=1).values

    
    print(df1['normality'].unique())
    print('y:',y)
    #smote = SMOTE(random_state=42)
    #X_res, y_res = smote.fit_resample(X, y)
    
       
    # Splitting training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)   
    
    text.insert(END, "\n\nTotal Records used for training: " + str(len(X_train)) + "\n")
    text.insert(END, "\n\nTotal Records used for testing: " + str(len(X_test)) + "\n\n")


    # Normalize feature columns except target columns
    for index in range(len(df1.columns) - 3):  # Exclude the last three columns (target columns)
        df1.iloc[:, index] = (df1.iloc[:, index] - df1.iloc[:, index].mean()) / df1.iloc[:, index].std()


    # Correlation heatmap
    plt.figure(figsize=(14, 14))
    sns.set(font_scale=1)
    sns.heatmap(df1.corr(), cmap='GnBu_r', annot=True, square=True, linewidths=.5)
    plt.title('Variable Correlation in Heatmap')
    plt.show()


def PerformanceMetrics(algorithm, testY, predict):
    # Calculate regression metrics
    mae = mean_absolute_error(testY, predict)
    mse = mean_squared_error(testY, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predict)

    print(f"{algorithm} Mean Absolute Error (MAE): {mae}")
    print(f"{algorithm} Mean Squared Error (MSE): {mse}")
    print(f"{algorithm} Root Mean Squared Error (RMSE): {rmse}")
    print(f"{algorithm} R-squared (R²): {r2}")
    
    text.insert(END, "Performance Metrics of " + str(algorithm) + "\n")
    text.insert(END, "Mean Absolute Error (MAE): " + str(mae) + "\n")
    text.insert(END, "Mean Squared Error (MSE): " + str(mse) + "\n")
    text.insert(END, "Root Mean Squared Error (RMSE): " + str(rmse) + "\n")
    text.insert(END, "R-squared (R²): " + str(r2) + "\n\n")
    
    # Prediction plot
    plt.figure(figsize=(10, 6))
    plt.scatter(testY, predict, color='blue', label='Predicted vs Actual')
    plt.plot([testY.min(), testY.max()], [testY.min(), testY.max()], color='red', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{algorithm} Predictions vs Actual')
    plt.legend()
    plt.grid()
    plt.show()

def RidgeClassifier():
    global ridge_clf, X_train, X_test, y_train, y_test
    global predict

    Classifier = 'model/RidgeClassifier.pkl'
    if os.path.exists(Classifier):
        # Load the trained model from the file
        ridge_clf = joblib.load(Classifier)
        print("Model loaded successfully.")
        predict = ridge_clf.predict(X_test)
        calculateMetrics("RidgeClassifier", predict, y_test.iloc[0])
    else:
        # Initialize and train the Ridge Classifier model
        ridge_clf = RidgeClassifier()
        ridge_clf.fit(X_train, y_train.iloc[0])
        # Save the trained model to a file
        joblib.dump(ridge_clf, Classifier) 
        print("Model saved successfully.")
        predict = ridge_clf.predict(X_test)
        calculateMetrics("RidgeClassifier", predict, y_test.iloc[0])
       
def RidgeRegressor():
    global ridge_reg, X_train, X_test, y_train, y_test
    global predict

    Regressor = 'model/RidgeRegressor.pkl'
    if os.path.exists(Regressor):
        # Load the trained model from the file
        ridge_reg = joblib.load(Regressor)
        print("Model loaded successfully.")
        predict = ridge_reg.predict(X_test)
        PerformanceMetrics("Ridge Regressor", predict, y_test)
    else:
        # Initialize and train the Ridge Regressor model
        ridge_reg = Ridge()
        ridge_reg.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(ridge_reg, Regressor) 
        print("Model saved successfully.")
        predict = ridge_reg.predict(X_test)
        PerformanceMetrics("Ridge Regressor", predict, y_test)
        
def ANN():
    global ann_model, rfr_model, X_train, X_test, y_train, y_test
    global predict, sc

    # Standardize the input features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Define file paths for models
    ANN_Model_Path = 'model/ANN_Regressor.h5'
    RFR_Model_Path = 'model/RFR_Model.pkl'

    # Check if pre-trained models exist
    if os.path.exists(ANN_Model_Path) and os.path.exists(RFR_Model_Path):
        # Load the trained ANN model
        ann_model = load_model(ANN_Model_Path)
        print("ANN Regressor model loaded successfully.")

        # Get features from ANN for RFR training
        X_train_features = ann_model.predict(X_train)
        X_test_features = ann_model.predict(X_test)

        # Load the trained RFR model
        rfr_model = joblib.load(RFR_Model_Path)
        print("Random Forest Regressor model loaded successfully.")

        # Make predictions
        predictions = rfr_model.predict(X_test_features)
        print('Random Forest Regressor model predicted:', predictions)
        print('y_test output:', y_test)
        PerformanceMetrics("Random Forest Regressor", predictions, y_test)

    else:
        # Build and train the ANN model if not already trained
        ann_model = Sequential()
        ann_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        ann_model.add(Dense(64, activation='relu'))
        ann_model.add(Dense(64, activation='relu'))
        ann_model.add(Dense(32, activation='relu'))
        ann_model.add(Dense(1))  # Single output for regression; adjust if predicting multiple values

        # Compile the ANN model
        ann_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Train the ANN model
        ann_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

        # Save the trained ANN model to a file
        ann_model.save(ANN_Model_Path)
        print("ANN Regressor model saved successfully.")

        # Get features from the ANN model for RFR
        X_train_features = ann_model.predict(X_train)
        X_test_features = ann_model.predict(X_test)

        # Train the Random Forest Regressor
        rfr_model = RandomForestRegressor()
        rfr_model.fit(X_train_features, y_train)

        # Save the trained RFR model to a file
        joblib.dump(rfr_model, RFR_Model_Path)
        print("Random Forest Regressor model saved successfully.")

        # Make predictions
        predictions = rfr_model.predict(X_test_features)
        print('Random Forest Regressor model predicted:', predictions)
        print('y_test output:', y_test)
        PerformanceMetrics("Random Forest Regressor", predictions, y_test)

         
def predict():
    global sc, rfr_model, ann_model, labels
    labels=[
    "normal",            # 0
    "wrong setup",       # 1
    "ddos",              # 2
    "Data type probing", # 3
    "scan attack",       # 4
    "man in the middle"  # 5
    ]
    
    # Load test file
    file = filedialog.askopenfilename(initialdir="Datasets")
    test = pd.read_csv(file)
    
    # Display loaded test data
    text.delete('1.0', END)
    text.insert(END, f'{file} Loaded\n')
    text.insert(END, "\n\nLoaded test data: \n" + str(test) + "\n")
    
    # Remove feature names to avoid StandardScaler issue
    test_values = test.values
    
    # Apply scaling
    test_scaled = sc.transform(test_values)
    
    # Make predictions using the ANN model
    ann_predictions = ann_model.predict(test_scaled)
    
    # RFC model uses the ANN predictions as input (verify this is intended)
    rfc_predictions = rfr_model.predict(ann_predictions)
    
    # If RFC gives multi-class probabilities, use argmax to get class indices
    #predicted_classes = rfc_predictions.argmax(axis=1)
    
    # Map predicted class indices to class labels
    #predicted_labels = [labels[p] for p in predicted_classes]
    
    # Add the predicted values to the test data
    test['Predicted'] = rfc_predictions
    
    # Display the predictions
    text.insert(END, "\n\nModel Predicted value in test data: \n" + str(test) + "\n")


  
def close():
  main.destroy()

# Main window setup
main = Tk()
main.title("IoT Network Performance Prediction")
main.geometry("1200x800")  # Spacious window size
main.config(bg='#2B3A67')  # Navy Blue background for a sleek look

# Title Label with a gradient-like dark-to-light theme
font = ('Verdana', 20, 'bold')
title = Label(main, text='ML-Driven Regressor for Enhancing IoT Network Performance Through Predictive Modeling',
              bg='#282828', fg='#FFD700', font=font, height=2)  # Dark background with Gold text
title.pack(fill=X, pady=10)

# Frame to hold buttons and text console
main_frame = Frame(main, bg='#2B3A67')  # Navy Blue for consistency
main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Frame to hold buttons (centered and in two rows)
button_frame = Frame(main_frame, bg='#2B3A67')  # Consistent background
button_frame.pack(pady=20)

# Button Font and Style
font1 = ('Arial', 12, 'bold')

# Helper function to create buttons with fancy color tones
def create_button(text, command, row, column):
    btn = Button(button_frame, text=text, command=command, bg='#1E90FF', fg='white',  # Dodger Blue buttons
                 activebackground='#FFA07A', font=font1, width=25, relief=RAISED, bd=4)  # Light Salmon hover effect
    btn.grid(row=row, column=column, padx=20, pady=15)

# Adding buttons in two rows, three buttons per row
create_button("Upload IoT Network Dataset", upload, 0, 0)
create_button("Data Preprocessing and EDA", preprocess, 0, 1)
create_button("Ridge Regressor", RidgeRegressor, 0, 2)
create_button("ANN+RF Regressor", ANN, 1, 0)
#create_button("Performance Metrics Graph", graph, 1, 1)
create_button("Prediction on Test Data", predict, 1, 1)

# Text console styling with scrollbar in fancy tones
text_frame = Frame(main_frame, bg='#2B3A67')  # Consistent background
text_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Text box styling with a centered and modern look
text = Text(text_frame, height=20, width=90, wrap=WORD, bg='white', fg='black', font=('Normal Text', 14))  # Wheat background
scroll = Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.pack(side=LEFT, fill=BOTH, expand=True)
scroll.pack(side=RIGHT, fill=Y)

# Adding the Close Application button with consistent style and size
close_button = Button(button_frame, text="Close Application", command=close, bg='#B22222', fg='white',  # Firebrick button
                      activebackground='#FF6347', font=font1, width=25, relief=RAISED, bd=4)

# Placing the Close button in the second row, third column (consistent layout)
close_button.grid(row=1, column=2, columnspan=3, padx=20, pady=15)


main.mainloop()