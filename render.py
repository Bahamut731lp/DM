import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load custom style from a local folder
plt.style.use('./rose-pine-dawn.mplstyle')

def gain_chart(classifier, x_test, y_test, drug_encoder, name):
    y_test_proba = classifier.predict_proba(x_test)
    class_labels = classifier.classes_

    # Plot
    plt.plot([0, 1], [0, 1], linestyle='--', label="Random", color="gray")

    plt.figure(figsize=(6, 4))    
    for i, class_idx in enumerate(class_labels):
        # Binary ground truth: 1 for this class, 0 otherwise
        y_true_bin = (y_test == class_idx).astype(int)
        y_score = y_test_proba[:, i]

        # Sort by predicted score descending
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true_bin.iloc[sorted_indices]

        drug_label = drug_encoder.inverse_transform([class_idx])[0]

        # Cumulative gain: how many positives we got after N samples
        cumulative_gains = np.cumsum(y_true_sorted) / y_true_bin.sum()
        percentages = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)

        plt.plot(percentages, cumulative_gains, label=drug_label)

    plt.title(f"Gains Chart – {name}")
    plt.xlabel("Percentage of Samples")
    plt.ylabel("Percentage of Positives Captured")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plt.savefig(f"./report/images/{name.lower().replace(' ', '_')}_gains.png")
    plt.close()

def scatter_plot(df: pd.DataFrame):
    # Najdi unikátní hodnoty léků a přiřaď jim barvy
    drugs = df["Drug"].unique()
    colors = plt.cm.tab10(range(len(drugs)))  # Vyber barvy z colormapy

    # Vytvoř scatter plot
    plt.figure(figsize=(8, 6))
    for drug in drugs:
        sub_df = df[df["Drug"] == drug]
        plt.scatter(sub_df["Na"], sub_df["K"], label=drug, s=80)

    plt.legend(title="Drug", loc="upper left", fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig("./report/images/scatter.png", transparent=True)
    
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