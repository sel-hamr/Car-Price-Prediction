# Car Price Prediction

This repository contains a machine learning project for predicting car prices based on various features such as make, model, year, fuel type, transmission, engine size, and mileage. The project uses a `RandomForestRegressor` model from the `scikit-learn` library.

## Project Structure

- `model.py`: Contains functions for loading data, transforming categorical features, training the model, and making predictions.
- `main.py`: Entry point for the application, handling command-line arguments and invoking training or prediction functions.
- `Car_Price.csv`: Example dataset file (not included in the repository).
- `README.md`: Project documentation.

## Requirements

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `pickle`

You can install the required packages using:
```sh
pip install pandas numpy scikit-learn
```

## Usage

### Training the Model

To train the model, run the following command:
```sh
python main.py -L data/Car_Price.csv
```
This will load the dataset, preprocess the data, train the `RandomForestRegressor` model, and save the model and transformer to `model.pkl` and `transformer.pkl` respectively.

### Making Predictions

To make predictions, simply run:
```sh
python main.py
```
You will be prompted to enter the car details interactively. The model will then predict the price based on the input features.

### Example

```sh
python main.py -L data/Car_Price.csv
python main.py
```

## Notes

- Ensure that the dataset file is in CSV format and contains the necessary columns: `Make`, `Model`, `Year`, `Fuel Type`, `Transmission`, `Engine Size`, `Mileage`, and `Price`.
- The same transformer used during training is applied during prediction to maintain consistency in the number of features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.