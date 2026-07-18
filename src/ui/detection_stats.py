import pandas as pd
from collections import Counter


def mean_detection_pie(detection_counts: list[Counter]):
    """
    Gather mean amount of objects of each type detected on image
    and create a pie chart from this data.

    Parameters
    -----
    detection_counts: Per-image detection counts

    Returns
    -----
    plt.Figure with a piechart
    """
    mean_counts = pd.DataFrame(detection_counts).fillna(0).mean().to_frame(name="count")
    total = mean_counts["count"].sum()
    pie_ax = mean_counts.plot.pie(
        y="count",
        autopct=lambda p: f"{int(round(p * total / 100))}",
        title="Average number of objects found per image",
        radius=0.7,
    )
    pie_ax.legend(loc=2, prop={"size": 6})
    return pie_ax.figure


def detection_trend_line(detection_counts: list[Counter]):
    """
    For each class, create a line plot of counts of detected
    objects. Draw plots for all classes on a single figure.

    Parameters
    -----
    detection_counts: Per-image detection counts

    Returns
    -----
    plt.Figure with a lineplot
    """
    df = pd.DataFrame(detection_counts).fillna(0)
    ax = df.plot.line()
    ax.set_xlabel("image no.")
    ax.set_ylabel("object count")
    return ax.figure


stats_to_show = [
    mean_detection_pie,
    detection_trend_line,
]
