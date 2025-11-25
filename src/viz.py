import seaborn as sns
import matplotlib.pyplot as plt
from .analysis import analyse_percentage, calc_popularity


def plot_percent_graph():
    percentage_df = analyse_percentage()
    # percentage_df = percentage_df.groupby("class_name").set_index("class_name")
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create grouped bar chart
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=percentage_df,
        x="class_name",
        y="percentage",
        hue="sentiment"
    )
    
    # Add labels and title
    plt.ylabel("Percentage (%)")
    plt.title("Sentiment Distribution by Class")
    plt.ylim(0, 100)
    plt.legend(title="Sentiments")
    plt.show()
    
    
    

def plot_popularity():
    
    grouped_df = calc_popularity()
    # Reset index so 'class' becomes a column
    df_long = grouped_df.reset_index().melt(
        id_vars="class_name",
        var_name="sentiment",
        value_name="count"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_long,
        x="class_name",
        y="count",
        hue="sentiment"
    )

    plt.xlabel("Class Name")
    plt.ylabel("Number of Sentiments")
    plt.title("Sentiment Distribution per Class")
    plt.tight_layout()
    plt.show()



plot_percent_graph()
plot_popularity()