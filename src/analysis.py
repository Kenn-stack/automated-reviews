from src.utils import authenticate
import pandas as pd 



def analyse_data():
    spreadsheet = authenticate()
    worksheet = spreadsheet.worksheet("processed")
    df = pd.DataFrame(worksheet.get_all_records())
    df["class_name"] = df["class_name"].astype("category")
    df_clean = df.dropna(subset=["class_name"])
    
    # percentage breakdow of sentiments per clothing class
    sentiment_counts = df_clean.groupby(["class_name", "sentiment"]).size().reset_index(name="count")
    class_totals = df_clean.groupby("class_name").size().reset_index(name="total")
    
    sentiment_summary = sentiment_counts.merge(class_totals, on="class_name")
    
    sentiment_summary["percentage"] = (sentiment_summary["count"] / sentiment_summary["total"]) * 100
    
    
    
    
    
    
    # df_clean["percentage"] = grouped["sentiment"].transform(lambda x: 100 * (x.count() / len(x)))
    
    # df.loc[:, "pct%"] = df.groupby(["class_name", "sentiment"]).value_counts(normalize=True).mul(100)
    print(sentiment_summary[["class_name", "sentiment", "percentage"]])
    
analyse_data()    