import logging
import json
import asyncio
import time
import random
import hashlib

from typing import cast

from asyncio import Semaphore
import groq
from groq import Groq, AsyncGroq
from gspread.spreadsheet import Spreadsheet
from gspread.worksheet import Worksheet
import numpy as np
import gspread
from gspread.exceptions import SpreadsheetNotFound, WorksheetNotFound
import pandas as pd
from pathlib import Path
from decouple import Config, RepositoryEnv 


SCRIPT_DIR = Path(__file__).resolve().parent

config = Config(repository=RepositoryEnv(SCRIPT_DIR.parent / ".env"))
api_key: str = cast(str, config("GROQ_API_KEY"))

client = AsyncGroq(
    api_key=api_key
)

system_prompt = """
      You are an expert review summarizer.  
      You will receive a list of customer reviews 
      from an e-commerce clothing store specifically for women's clothing.

      TASKS FOR EACH REVIEW:
      1. Generate a ONE-SENTENCE summary capturing the essential meaning.
      2. Assign sentiment: "positive", "negative", or "neutral".
      3. If a review is extremely short or cannot be summarized,
      return the original text as the summary.

      RETURN FORMAT (STRICT):
      Return ONLY a valid JSON list where each item is:
      {
      "summary": "...",
      "sentiment": "positive|negative|neutral"
      }
"""

logger = logging.getLogger(__name__)


sem = asyncio.Semaphore(5)

def authenticate():
    """
    Authenticates gspread
    
    Args:
        
        
    Returns:
        Client: a gspread client
    """
    
    service_account_file_path = SCRIPT_DIR.parent / "service-account-key.json"
    
    gc = gspread.service_account(filename=service_account_file_path)
    spreadsheet = gc.open("Data-Epic")
    return spreadsheet
    
    

def df_hash(df: pd.DataFrame) -> str:
    # Normalize: sort rows & columns, reset index, convert to string
    normalized = df.sort_index(axis=1).sort_values(list(df.columns)).reset_index(drop=True)
    data_str = normalized.to_csv(index=False)
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    
def check_idempotence(spreadsheet):
    """
    Checks if data to be run already exists in the spreadsheet.
    
    Args:
        spreadsheet: An instance of Spreadsheet
        
    Returns:
        True: pipeline should be run
        False: skip
        
    """

    try:
        csv_path = SCRIPT_DIR.parent / "data" / "reviews.csv"
        logger.info("Reading CSV into pandas dataframe")
        df = pd.read_csv(csv_path)
        raw_df = df.iloc[:200]
        raw_df = raw_df.replace([np.inf, -np.inf], np.nan).fillna("")
        # raw_df = raw_df.astype(object).where(raw_df.notnull(), None)
    
        
        raw_df_hash = df_hash(raw_df)
        worksheet = spreadsheet.worksheet("raw")
        worksheet_df_hash = df_hash(pd.DataFrame(worksheet.get_all_records()))
        
        if raw_df_hash == worksheet_df_hash:
            return False
        else:
            return True
        
    except FileNotFoundError as e:
        logger.error(f"File Not Found: {str(e)}")
        raise Exception({
            "status": "error",
            "message": "An error occurred in opening the csv file.",
            "error": str(e)})
        
    except WorksheetNotFound:
        return True


def check_sheet_exists(spreadsheet: Spreadsheet, sheet: str):
    if sheet not in [ws.title for ws in spreadsheet.worksheets()]:
        worksheet = spreadsheet.add_worksheet(title=sheet, rows=300, cols=20)  
    else:
        worksheet = spreadsheet.worksheet(sheet)
    
    return worksheet
    

def load_raw(spreadsheet: Spreadsheet):
    """
    Loads the raw data from the csv file to a sheet named 'raw'.
    Data should contain only 200 lines.
    
    Args:
        spreadsheet: An instance of Spreadsheet
        
    Returns:
        Worksheet: the "raw" sheet in my spreadsheet
    """
    try:
        csv_path = SCRIPT_DIR.parent / "data" / "reviews.csv"
        logger.info("Reading CSV into pandas dataframe")
        df = pd.read_csv(csv_path)
        raw_df = df.iloc[:200]
        raw_df = raw_df.replace([np.inf, -np.inf], np.nan)
        raw_df = raw_df.astype(object).where(raw_df.notnull(), None)
        
        
        worksheet = check_sheet_exists(spreadsheet, "raw")
        worksheet.update([raw_df.columns.values.tolist()] + raw_df.values.tolist())
        
        
    except FileNotFoundError as e:
        logger.error(f"File Not Found: {str(e)}")
        raise Exception({
            "status": "error",
            "message": "An error occurred in opening the csv file.",
            "error": str(e)})
        
    except SpreadsheetNotFound as e :
        logger.error(f"Spreadsheet not found: {str(e)}")
        raise Exception({
            "status": "error",
            "message": "An error occurred in opening the spreadsheet.",
            "error": str(e)})
        
    logger.info({
        "status": "sucess",
        "message": "CSV data successfully loaded to worksheet 'raw'"
    })
    return worksheet
    
   
   
def clean_data(spreadsheet: Spreadsheet, worksheet: Worksheet):
    """
    Loads data into pandas dataframe, cleans it and saves it another worksheet.
    
    Args:
        spreadsheet: An instance of Spreadsheet
        worksheeet: An instance of Worksheet
        
    Returns:
        
    """
    raw_df = pd.DataFrame(worksheet.get_all_records())
    
    #drop unnecessary column
    raw_df = raw_df.drop('Unnamed: 0', axis=1)
    
    # normalize column names and set type of non-integer column to string
    raw_df.columns = [col.strip().replace(" ", "_").lower() for col in raw_df.columns]
    str_cols = ['title', 'review_text', 'class_name', 'division_name', 'department_name']
    raw_df[str_cols] = raw_df[str_cols].astype("string")
   
    # replace missing values with empty string
    raw_df[['title', 'review_text']] =  raw_df[['title', 'review_text']].fillna('')
    
    # correct misspelt words
    raw_df = raw_df.replace(['Initmates', 'Intimate'], 'Intimates')

    # set type of dress categories to category
    # cat_cols = ['division_name', 'department_name', 'class_name']
    # raw_df[cat_cols] = raw_df[cat_cols].astype("category")

    # save dataframe to sheet
    if "staging" not in [ws.title for ws in spreadsheet.worksheets()]:
         worksheet = spreadsheet.add_worksheet(title="staging", rows=300, cols=20)
         worksheet.update([raw_df.columns.values.tolist()] + raw_df.values.tolist())
         return raw_df
         
    worksheet = spreadsheet.worksheet("staging")
    worksheet.update([raw_df.columns.values.tolist()] + raw_df.values.tolist())
    
    logger.info({
        "status": "sucess",
        "message": "DataFrame successfully loaded to worksheet 'staging'"
    }) 
    
    return raw_df
    
    
def summarization(  
                    df: pd.DataFrame, 
                    spreadsheet: Spreadsheet,
                    review_column: str, 
                    summary_column: str ="summary", 
                    sentiment_column: str ="sentiment"):
    """
    Summarizes customer reviews in batches of 10 using Groq API.
    Adds two new columns: summary + sentiment.
    """
    
    client = Groq(
        api_key=api_key
    )

    # Prepare empty columns
    df[summary_column] = None
    df[sentiment_column] = None
    
    # df = df.iloc[:100]

    # Batch size
    batch_size = 10

    # Optimized prompt template
    system_prompt = """
        You are an expert review summarizer.  
        You will receive a list of customer reviews 
        from an e-commerce clothing store specifically for women's clothing.

        TASKS FOR EACH REVIEW:
        1. Generate a ONE-SENTENCE summary capturing the essential meaning.
        2. Assign sentiment: "positive", "negative", or "neutral".
        3. If a review is extremely short or cannot be summarized,
        return the original text as the summary.

        RETURN FORMAT (STRICT):
        Return ONLY a valid JSON list where each item is:
        {
        "summary": "...",
        "sentiment": "positive|negative|neutral"
        }
        """

    # combined_records = []
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        #check for case of nan or empty reviews
        valid_reviews_df = batch[batch[review_column].notnull() & (batch[review_column].str.strip() != '')]

        # null_reviews_df = batch[batch[review_column].isnull() | (batch[review_column].str.strip() == '')]
        valid_reviews_list = valid_reviews_df[review_column].tolist()

        # Create batch prompt
        user_prompt = f"Here are the reviews:\n{json.dumps(valid_reviews_list, indent=2)}"

        # Call API
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            temperature=0.4,
            max_completion_tokens=2048,
            top_p=1,
        )

        # print("response", response)

        # Extract generated JSON string
        raw_output = response.choices[0].message.content
        strip_output = raw_output.strip()

        # Convert JSON list → Python list
        try:
            results = json.loads(strip_output)
            # print("results", results)
        except json.JSONDecodeError:
            logger.error({
                "status": "error",
                "message": f"JSON parse error for batch starting at index {i}",
                "raw_output": raw_output
            }
            )
            raise Exception({
                "status": "error",
                "message": f"JSON parse error for batch starting at index {i}",
                "raw_output": raw_output
            })

        # Insert results back into df
        for idx, result in zip(batch.index, results):
            valid_reviews_df.at[idx, summary_column] = result.get("summary")
            valid_reviews_df.at[idx, sentiment_column] = result.get("sentiment")
            df = df.combine_first(valid_reviews_df)
            
    df.loc[:, "action_needed"] = df["sentiment"].apply(lambda x: "Yes" if x == "negative" else "No")
    df = df.replace([np.inf, -np.inf], np.nan).fillna("")

    
    worksheet = check_sheet_exists(spreadsheet, "processed")
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

            
    return df
        
      
      
# def batch(df, batch_size, review_column: str="review_text"):
#     for i in range(0, len(df), batch_size):
#         batch = df.iloc[i : i + batch_size]
#         #check for case of nan or empty reviews
#         valid_reviews_df = batch[batch[review_column].notnull() & (batch[review_column].str.strip() != '')]
        
#         valid_reviews_list = valid_reviews_df[review_column].tolist()
#         valid_index_list = valid_reviews_df.index.tolist()

        
#         yield {
#                 "valid_reviews_list": valid_reviews_list,
#                 "valid_index_list": valid_index_list
#                 }
                
# async def call_groq_with_retry(*args, **kwargs):
#     max_retries = 5
#     for _ in range(max_retries):
#         try:
#             return await client.chat.completions.create(*args, **kwargs)
#         except groq.RateLimitError:
#             wait = 7 + random.random()  # avoid thundering herd
#             print(f"Rate limited. Waiting {wait:.2f}s...")
#             await asyncio.sleep(wait)

#     raise Exception("Max retries exceeded")
        
       
# async def summarize_batch(batch):
#     user_prompt = f"Here is the review:\n{json.dumps(batch, indent=2)}"
    
#     async with sem:
#         response = await client.chat.completions.create(
#             model="openai/gpt-oss-120b",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user",    "content": user_prompt},
#             ],
#             temperature=0.4,
#             max_completion_tokens=20,
#             top_p=1,
#         )


#     # Extract generated JSON string
#     raw_output = response.choices[0].message.content
#     strip_output = raw_output.strip()
    
#     try:
#         results = json.loads(strip_output)
#         # print("results", results)
#     except json.JSONDecodeError:
#         logger.error({
#             "status": "error",
#             "message": "JSON parse error for batch",
#             "raw_output": raw_output
#         }
#         )
#         raise Exception({
#             "status": "error",
#             "message": "JSON parse error for batch",
#             "raw_output": raw_output
#         })
        
#     return results
    

          
    
# async def async_summarization(  
#                     df: pd.DataFrame, 
#                     spreadsheet: Spreadsheet):
#     """
#     Summarizes customer reviews in batches of 10 using Groq API.
#     Adds two new columns: summary + sentiment.
#     """
    
#     # Prepare empty columns
#     df["summary"] = None
#     df["sentiment"] = None
    
#     # df = df.iloc[:100].copy()

#     # Batch size
#     batch_size = 10


#     # Process in batches
#     valid_reviews_lists = [b.get("valid_reviews_list") for b in batch(df, batch_size)]
#     valid_index_lists = [b.get("valid_index_list") for b in batch(df, batch_size)]
#     tasks = [asyncio.create_task(summarize_batch(b)) for b in valid_reviews_lists]
#     # results = []
#     # for i in range(0, len(tasks), 2):
#     #     results.append(await asyncio.gather(*tasks[i:i+2]))
#     results = await asyncio.gather(*tasks)
#     print(len(results))
    
    
#     flattened_summary = [item.get("summary") for sublist in results for item in sublist]
#     flattened_sentiment = [item.get("sentiment") for sublist in results for item in sublist]
#     flattened_index = [item for sublist in valid_index_lists for item in sublist]

#     df.loc[flattened_index, "summary"] = flattened_summary
#     df.loc[flattened_index, "sentiment"] = flattened_sentiment
    

#     df.loc[:, "action_needed"] = np.where(df["sentiment"] == "positive", "No", "Yes")
    
#     worksheet = check_sheet_exists(spreadsheet, "processed")
#     worksheet.update([df.columns.values.tolist()] + df.values.tolist())
       
#     return df

    
    



 
# sh = authenticate()
# # sh = create_spreadsheet(gc)
# raw_ws = load_raw(sh)
# stag_df = clean_data(sh, raw_ws)

# start = time.perf_counter()
# summ_df = summarization(stag_df, sh, "review_text")
# end= time.perf_counter()
# print(f"Sync_time: {end - start:.4f} seconds")
# print(summ_df)


# async def main():
#     start = time.perf_counter()
#     result = await async_summarization(stag_df, sh)
#     end= time.perf_counter()
#     print(f"Async_time: {end - start:.4f} seconds")
#     print(result)
# asyncio.run(main())   

