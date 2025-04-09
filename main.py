import json
import logging
import logging.config
import questionary

import train
import predict

LOGGING_CONFIG = "./logging.json"
INPUT_DATA = "./data/DRUG1n"
TESTING_DATA = "./data/DRUG1n_test"

def setLoggingConfig(filename: str):
    with open(filename, 'r') as f:
        config = json.load(f)
        logging.config.dictConfig(config)

if __name__ == "__main__":
    setLoggingConfig(LOGGING_CONFIG)

    while True:
        action = questionary.select(
            "What do you want to do?",
            choices=[
                questionary.Choice("Train", "train"),
                questionary.Choice("Predict", "predict"),
                questionary.Choice("Exit", "exit")
            ],
        ).ask()

        if action == "train":
            train.start()

        if action == "predict":
            predict.start()

        if action == "exit":
            break
