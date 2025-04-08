import csv
import json
import logging
import logging.config
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, log_loss

import render
from report import Report

LOGGING_CONFIG = "./logging.json"
INPUT_DATA = "./data/DRUG1n"
TESTING_DATA = "./data/DRUG1n_test"

def setLoggingConfig(filename: str):
    with open(filename, 'r') as f:
        config = json.load(f)
        logging.config.dictConfig(config)

def getDataFromFile(filename: str):
    rows = []

    with open(filename, encoding="utf-8") as file_handle:
        spamreader = csv.DictReader(file_handle, delimiter=",")

        for row in spamreader:
            rows.append(row)

    logging.info("Loaded %d rows from %s", len(rows), filename)

    return rows

def getImportantPredictors(data):
    # Encode categorical columns to numerical
    logging.info("Computing feature importance (Mutual Information Algorithm).")
    logging.debug("Transforming categorical data to numerical.")
    encoded_data = data.copy()
    for col in encoded_data.select_dtypes(include="category").columns:
        encoded_data[col] = LabelEncoder().fit_transform(encoded_data[col])

    logging.debug("Splitting variables to predictors and responses.")
    x = encoded_data.drop(columns=["Drug"])
    y = encoded_data["Drug"]

    # Compute mutual information
    logging.debug("Computing mutual information estimation.")
    mi_scores = mutual_info_classif(x, y, discrete_features='auto')

    # Create a DataFrame for easier sorting/plotting
    logging.debug("Creating a DataFrame for easier sorting/plotting.")
    mi_df = pd.DataFrame({"Feature": x.columns, "Score": mi_scores})
    mi_df = mi_df.sort_values(by="Score", ascending=True)

    # Plot using matplotlib
    logging.debug("Creating a feature importance graph.")
    render.feature_importance(mi_df["Feature"], mi_df["Score"])

    logging.info("Dropping unimportant features.")
    return mi_df[mi_df["Score"] > 0.1]["Feature"].to_list()

def getPreprocessedData(data: pd.DataFrame):
    type_mappings = {
        "Sex": "category",
        "BP": "category",
        "Cholesterol": "category"
    }

    # Retyping infered objects into more specific dtypes
    for column, dtype in type_mappings.items():
        logging.debug("Retyping column '%s' from %s to %s.", column, data[column].dtype, dtype)
        data[column] = data[column].astype(dtype)

    if "Na_to_K" not in data.columns.to_list():
        data["Na_to_K"] = data["Na"] / data["K"]
        data = data.drop(columns=["Na", "K"])

    # Scaling for MLP
    data["Na_to_K"] = StandardScaler().fit_transform(data[["Na_to_K"]])

    # Encode categorical data
    encoder = LabelEncoder()
    categorical_predictors = data.select_dtypes(include=["category"]).columns
    for col in categorical_predictors:
        data[col] = encoder.fit_transform(data[col])

    return data

def main():
    # Configure logging with configuration from JSON file.
    setLoggingConfig(LOGGING_CONFIG)

    # Read data from CSV
    logging.info("Reading input data from %s", INPUT_DATA)
    training_data = pd.read_csv(INPUT_DATA)
    testing_data = pd.read_csv(TESTING_DATA)

    logging.debug("Loaded training data with %d rows with %d columns.", len(training_data.index), len(training_data.columns))
    logging.debug("Loaded testing data with %d rows with %d columns.", len(testing_data.index), len(testing_data.columns))

    # List of classifiers to evaluate
    classifiers = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Multilayer Perceptron': MLPClassifier(max_iter=1000, learning_rate_init=0.001),
        'Decision Tree': DecisionTreeClassifier()
    }

    # Labels to class numbers
    drug_encoder = LabelEncoder()
    drug_encoder.fit(pd.concat([training_data["Drug"], testing_data["Drug"]], axis=0))

    training_data["Drug"] = drug_encoder.transform(training_data["Drug"])
    testing_data["Drug"] = drug_encoder.transform(testing_data["Drug"])

    training_data = getPreprocessedData(training_data)
    testing_data = getPreprocessedData(testing_data)

    # Getting predictors and target variables
    predictors = getImportantPredictors(training_data)
    targets = "Drug"

    # Copying for immutability
    x = training_data[predictors].copy()
    y = training_data[targets].copy()

    # Split dataset to training set and validation set
    logging.info("Splitting dataset to training and validation sets.")
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_test = testing_data[predictors].copy()
    y_test = testing_data[targets].copy()

    logging.debug("Rendering relative data split.")
    render.data_split_relative(len(x_train), len(x_validation), len(x_test))

    logging.debug("Rendering absolute data split.")
    render.data_split_absolute(len(x_train), len(x_validation), len(x_test))
    
    logging.debug("Rendering target variable distribution.")
    render.class_distribution(pd.DataFrame(drug_encoder.inverse_transform(pd.DataFrame(y_train)["Drug"])).value_counts())

    # Evaluate each model using cross-validation
    models = []
    report = Report("./report/report.html")

    for name, classifier in classifiers.items():
        logging.debug("Training %s: Started,", name)
        classifier.fit(x_train, y_train)

        logging.debug("Training %s: Running prediction on validation and test sets.", name)
        validation_prediction = classifier.predict(x_validation)
        test_prediction = classifier.predict(x_test)

        logging.debug("Training %s: Creating classification reports.", name)
        validation_report = classification_report(y_validation, validation_prediction, output_dict=True, zero_division=1)
        testing_report = classification_report(y_test, test_prediction, output_dict=True, zero_division=1)

        validation = pd.DataFrame(validation_report).transpose()
        testing = pd.DataFrame(testing_report).transpose()

        validation.index = validation.index.map(lambda x: drug_encoder.inverse_transform([int(x)])[0] if x.isdigit() else x)
        testing.index = testing.index.map(lambda x: drug_encoder.inverse_transform([int(x)])[0] if x.isdigit() else x)

        logging.debug("Training %s: Computing accuracy.", name)
        validation_acc = max(validation.loc["accuracy"])
        testing_acc = max(testing.loc["accuracy"])

        validation = validation[validation.index.str.startswith("drug")].copy()
        testing = testing[testing.index.str.startswith("drug")].copy()

        logging.debug("Training %s: Appending to results", name)
        models.append({
            "name": name,
            "validation": {
                "acc": validation_acc
            },
            "testing": {
                "acc": testing_acc
            },
            "stats": pd.merge(validation, testing, left_index=True, right_index=True, suffixes=('_val', '_test'))
        })

        logging.debug("Training %s: Ended.", name)
    
    models = sorted(models, key=lambda x: x.get("testing", {}).get("acc", 0), reverse=True)

    for model in models:
        report.add_model_result(model)

    report.end()

if __name__ == "__main__":
    main()
