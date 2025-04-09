import logging
import pathlib
import pickle
import questionary
import pandas as pd

from sklearn.calibration import LabelEncoder

def start():
    models = pathlib.Path("./models").glob('**/*')
    models = [x.stem for x in models if x.is_file()]
    models = [questionary.Choice(x.capitalize().replace("_", " "), x) for x in models]

    classifier = None
    encoder_path = pathlib.Path("./encoders/drug_encoder.pkl")
    drug_encoder: LabelEncoder = None

    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            drug_encoder = pickle.load(f)
    else:
        logging.critical("No encoder was found at %s, try training some models first.", encoder_path.as_posix())
        return

    model = questionary.select(
        "What model you want to use?",
        choices=models,
    ).ask()

    model_path = pathlib.Path(f"./models/{model}.pkl")
    logging.debug("Loading model from %s", model_path.as_posix())

    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    
    logging.debug("%s has been loaded.", model_path)

    bp: str = questionary.select("Blood Pressure", [
        questionary.Choice("Low", 0),
        questionary.Choice("Normal", 1),
        questionary.Choice("High", 2)
    ]).ask()

    while True:
        sodium: str = questionary.text("Sodium level").ask()

        try:
            sodium = float(sodium)
            break
        except ValueError:
            logging.info("Sodium level %s is not a valid floating point number.", sodium)

    while True:
        potassium: str = questionary.text("Potassium level").ask()
        try:
            potassium = float(potassium)
            break
        except ValueError:
            logging.info("Potassium level %s is not a valid floating point number.", potassium)

    data = pd.DataFrame([[bp, float(sodium) / float(potassium)]], columns=["BP", "Na_to_K"])
    prediction = classifier.predict(data)
    drug_name = drug_encoder.inverse_transform(prediction)
    
    logging.info("Predicting drug for patient")
    logging.info("Patient should be prescribed %s", drug_name[0])
    return