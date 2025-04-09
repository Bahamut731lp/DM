import logging
import os
import pickle
import pathlib
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import render
from report import Report

LOGGING_CONFIG = "./logging.json"
INPUT_DATA = "./data/DRUG1n"
TESTING_DATA = "./data/DRUG1n_test"

def get_predictors(data: pd.DataFrame):
    """Function to return predictors relevant for training models.

    Parameters
    ----------
    data
        Dataframe with training data

    Returns
    -------
        Array of predictors
    """

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


def get_preprocessed_data(data: pd.DataFrame):
    """Function that retypes columns into more specific dtypes

    Parameters
    ----------
    data
        Dataframe

    Returns
    -------
        Retyped dataframe
    """

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
    #data["Na_to_K"] = StandardScaler().fit_transform(data[["Na_to_K"]])

    # Encode categorical data
    encoder = LabelEncoder()
    categorical_predictors = data.select_dtypes(include=["category"]).columns
    for col in categorical_predictors:
        data[col] = encoder.fit_transform(data[col])

    return data

def start():
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
        'Multilayer Perceptron': MLPClassifier(max_iter=7500, learning_rate_init=0.001),
        'Decision Tree': DecisionTreeClassifier()
    }

    # Labels to class numbers
    os.makedirs("./encoders", 666, True)
    encoder_path = pathlib.Path("./encoders/drug_encoder.pkl")
    drug_encoder: LabelEncoder = None

    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            drug_encoder = pickle.load(f)
    else:
        drug_encoder = LabelEncoder()
        drug_encoder.fit(pd.concat([training_data["Drug"], testing_data["Drug"]], axis=0))
        with open(encoder_path, "wb") as f:
            pickle.dump(drug_encoder, f, protocol=5)

    training_data["Drug"] = drug_encoder.transform(training_data["Drug"])
    testing_data["Drug"] = drug_encoder.transform(testing_data["Drug"])

    training_data = get_preprocessed_data(training_data)
    testing_data = get_preprocessed_data(testing_data)

    # Getting predictors and target variables
    predictors = get_predictors(training_data)
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
    inversed_classes = drug_encoder.inverse_transform(pd.DataFrame(y_train)["Drug"])
    classes_counts = pd.DataFrame(inversed_classes).value_counts()
    render.class_distribution(classes_counts)

    # Evaluate each model using cross-validation
    models = []
    report = Report("./report/report.html")
    os.makedirs("./models", 666, True)

    for name, classifier in classifiers.items():
        # Train model
        logging.debug("Training %s: Started.", name)
        classifier.fit(x_train, y_train)

        # Create predictions for validation and testing datasets
        logging.debug("Training %s: Running prediction on validation and test sets.", name)
        validation_prediction = classifier.predict(x_validation)
        test_prediction = classifier.predict(x_test)

        # Compute precision, recall and f1
        logging.debug("Training %s: Creating classification reports.", name)
        validation_report = classification_report(y_validation, validation_prediction, output_dict=True, zero_division=1)
        testing_report = classification_report(y_test, test_prediction, output_dict=True, zero_division=1)

        # Create DataFrames
        validation = pd.DataFrame(validation_report).transpose()
        testing = pd.DataFrame(testing_report).transpose()

        # Convert numeric classes back to original labels (i.e. 0 -> drugA)
        validation.index = validation.index.map(lambda x: drug_encoder.inverse_transform([int(x)])[0] if x.isdigit() else x)
        testing.index = testing.index.map(lambda x: drug_encoder.inverse_transform([int(x)])[0] if x.isdigit() else x)

        # Extract accuracy
        logging.debug("Training %s: Computing accuracy.", name)
        validation_acc = max(validation.loc["accuracy"])
        testing_acc = max(testing.loc["accuracy"])

        # Remove averages
        validation = validation[validation.index.str.startswith("drug")].copy()
        testing = testing[testing.index.str.startswith("drug")].copy()

        # Add model stats to results
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

        with open(f"./models/{name.lower().replace(' ', '_')}.pkl", "wb") as f:
            pickle.dump(classifier, f, protocol=5)

        if isinstance(classifier, DecisionTreeClassifier):
            render.tree(classifier, drug_encoder)

        logging.debug("Training %s: Ended.", name)

    # Sort models by their accuracy on testing dataset
    models = sorted(models, key=lambda x: x.get("testing", {}).get("acc", 0), reverse=True)

    # Add sorted models to HTML report
    for model in models:
        report.add_model_result(model)

    report.end()