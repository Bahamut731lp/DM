import csv
import json
import logging
import logging.config

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

if __name__ == "__main__":
    setLoggingConfig("./logging.json")
    data = getDataFromFile("./data/DRUG1n")