import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

def load(path: str) -> pd.DataFrame:
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("The file fromat is not .csv")
        if not os.path.exists(path):
            raise AssertionError("The file doesn't exist")
    except AssertionError as error:
        print(error)
    else:
      data = pd.read_csv(path)
      print(f"Loading dataset of dimension {data.shape}")
      return data

def turn_category_to_number(x_data: pd.DataFrame, transformer=None):
    categorical_features = ["Make", "Model", "Fuel Type", "Transmission"]
    if transformer is None:
        onehot = OneHotEncoder()
        transformer = ColumnTransformer([("onehot", onehot, categorical_features)], remainder="passthrough")
        return transformer.fit_transform(x_data), transformer
    else:
        return transformer.transform(x_data)

def train(path: str) -> None:
    try:
        print("Importing data...")
        data = load(path)
        data.dropna(subset=["Price"], inplace=True)
        print("Training model...")
        x = data.drop("Price", axis=1)
        y = data["Price"]
        x_transformed, transformer = turn_category_to_number(x)
        X_train, X_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        print(f"Model trained with accuracy: {model.score(X_test, y_test)}")
        with open("model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        with open("transformer.pkl", "wb") as transformer_file:
            pickle.dump(transformer, transformer_file)
        print("Model and transformer saved as model.pkl and transformer.pkl")

    except Exception as error:
        print(f"Error: {error}")

def predict() -> None:
    try:
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("transformer.pkl", "rb") as transformer_file:
            transformer = pickle.load(transformer_file)
        if model is None or transformer is None:
            raise AssertionError("Model or transformer not found try to train model first" )
        make = input("Enter the make of the car:  ")
        model_car = input("Enter the model of the car example ('Model B, Model A, etc ...'): ")
        year = int(input("Enter the year of the car: "))
        fuel_type = input("Enter the fuel type of the car: ")
        transmission = input("Enter the transmission of the car: ")
        engine_size = float(input("Enter the engine size of the car: "))
        mileage = int(input("Enter the mileage of the car: "))
        item_array = np.array([make, model_car, year, fuel_type, transmission, engine_size, mileage])
        df = pd.DataFrame([item_array],
                          columns=['Make', 'Model', 'Year', 'Fuel Type', 'Transmission', 'Engine Size', 'Mileage'])
        df_transformed = turn_category_to_number(df, transformer)
        prediction = model.predict(df_transformed)
        print(f"The predicted price of the car is: {'%.0f' % prediction[0]} Doller")

    except Exception as error:
        print(f"Error: {error}")
