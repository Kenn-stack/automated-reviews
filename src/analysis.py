from src.utils import authenticate
import pandas as pd 



def get_processed_as_df():
    spreadsheet = authenticate()
    worksheet = spreadsheet.worksheet("processed")
    df = pd.DataFrame(worksheet.get_all_records())
    # df = df[df["sentiment"] != ""]
    
    return df


def analyse_percentage():
    df = get_processed_as_df()
    df_clean = df[df["sentiment"] != '']
    
    # percentage breakdown of sentiments per clothing class
    group = df_clean.groupby(["class_name", "sentiment"]).size()
    total = df_clean.groupby("class_name").size()
    
    percentage = (group / total) * 100
    
    percentage_df = percentage.reset_index(name="percentage")
    
    
    return percentage_df
    

def calc_popularity():
    df = get_processed_as_df()

    grouped_df = df.groupby('class_name').agg(
        positives=("sentiment", lambda x: (x=='positive').sum()),
        negatives=("sentiment", lambda x: (x=='negative').sum()),
        neutral=("sentiment", lambda x: (x=='neutral').sum())
    )
    
    return grouped_df
    
    
def sentiment_distribution():
    df = get_processed_as_df()
    df_clean = df[df["sentiment"] != '']
    length = len(df_clean)
  
    
    df_clean = df_clean.groupby("sentiment").agg(
        count=("sentiment", lambda x: x.count()),
        percentage=("sentiment", lambda x: f"{((x.count() / length) * 100):.2f}")
    )
    
    return df_clean
    
  
print(sentiment_distribution())