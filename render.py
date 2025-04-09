import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load custom style from a local folder
plt.style.use('./rose-pine-dawn.mplstyle')

def tree(classifier: DecisionTreeClassifier, encoder: LabelEncoder):
    plt.figure(figsize=(60, 40))
    plot_tree(classifier, class_names=encoder.classes_)
    plt.tight_layout()
    plt.savefig("./report/images/decision_tree.png", transparent=True)

def feature_importance(feature, score):
    # Plot using matplotlib
    plt.figure(figsize=(8, 5))
    plt.barh(feature, score)
    plt.xlabel("Skóre vzájemné informace")
    plt.title("Důležitost prediktoru")
    plt.tight_layout()
    plt.savefig("./report/images/feature_importance.png", transparent=True)

def data_split_relative(training_size: int, validation_size: int, testing_size: int):
    plt.figure()
    labels = ["Trénovací data", "Validační data", "Testovací data"]
    values = [training_size, validation_size, testing_size]

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'white'}, pctdistance=0.85
    )

    # Make it a donut by adding a white circle in the center
    center_circle = plt.Circle((0,0), 0.70, fc='white')
    ax.add_artist(center_circle)

    # Set transparent background
    fig.patch.set_alpha(0)  # Transparent background for the whole figure
    ax.set_facecolor('none')  # Transparent axes

    plt.title("Relativní rozdělení dat")
    plt.savefig("./report/images/training_data_split.png", transparent=True)

def data_split_absolute(training_size: int, validation_size: int, testing_size: int):
    fig = plt.figure()
    ax = fig.add_subplot(111) 

    labels = ["Trénovací data", "Validační data", "Testovací data"]
    values = [training_size, validation_size, testing_size]

    # Create the pie chart
    bars = ax.bar(labels, values)
    ax.bar_label(bars, label_type="edge")

    fig.patch.set_alpha(0)  # Transparent background for the whole figure

    plt.title("Absolutní rozdělení dat")
    plt.savefig("./report/images/training_data_histogram.png", transparent=True)

def class_distribution(df: pd.DataFrame):
    fig = plt.figure()
    ax = fig.add_subplot(111) 

    labels = df.index.get_level_values(0).to_list()
    values = df.values
    
    # Create the pie chart
    bars = ax.bar(labels, values)
    ax.bar_label(bars, label_type="edge")

    # Create the pie chart
    fig.patch.set_alpha(0)  # Transparent background for the whole figure

    plt.title("Rozložení cílové proměnné v trénovacím datasetu")
    plt.savefig("./report/images/class_distribution.png", transparent=True)