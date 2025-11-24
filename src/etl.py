from .utils import load_raw, clean_data, summarization




def run_etl(spreadsheet):
    raw_worksheet = load_raw(spreadsheet)
    clean_df = clean_data(spreadsheet, raw_worksheet)
    summarization(clean_df, spreadsheet, "review_text")
    return {"message": "Success"}
    