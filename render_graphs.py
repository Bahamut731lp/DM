import matplotlib.pyplot as plt

# Load custom style from a local folder
plt.style.use('./rose-pine-dawn.mplstyle')


def render_data_split_graph():
    plt.figure()
    labels = ["Trénovací data", "Validační data", "Testovací data"]
    values = [230, 50, 80]

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

    plt.tight_layout()
    plt.savefig("./report/images/training_data_split.png", transparent=True)

def render_data_split_histogram():
    fig = plt.figure()
    ax = fig.add_subplot(111) 

    labels = ["Trénovací data", "Validační data", "Testovací data"]
    values = [230, 50, 80]

    # Create the pie chart
    bars = ax.bar(labels, values)
    ax.bar_label(bars, label_type="edge")

    fig.patch.set_alpha(0)  # Transparent background for the whole figure

    fig.tight_layout()
    fig.savefig("./report/images/training_data_histogram.png", transparent=True)

render_data_split_graph()
render_data_split_histogram()