import pytest
import pandas as pd
import json
from unittest.mock import MagicMock, patch
from pathlib import Path


# Import functions from your actual module
from src.utils import (
    df_hash,
    check_sheet_exists,
    load_raw,
    clean_data,
)


def test_df_hash_consistency():
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"b": ["y", "x"], "a": [2, 1]})  

    assert df_hash(df1) == df_hash(df2)




def test_check_sheet_exists_returns_existing():
    fake_ws = MagicMock(title="existing")
    spreadsheet = MagicMock()
    spreadsheet.worksheets.return_value = [fake_ws]

    ws = check_sheet_exists(spreadsheet, "existing")

    spreadsheet.add_worksheet.assert_not_called()
    assert ws.title == "existing"


@patch("pandas.read_csv")
def test_load_raw_updates_sheet(mock_read_csv):
    df = pd.DataFrame({"col": [1, 2]})
    mock_read_csv.return_value = df

    spreadsheet = MagicMock()
    fake_ws = MagicMock()
    spreadsheet.worksheet.side_effect = Exception("not found")
    spreadsheet.add_worksheet.return_value = fake_ws

    ws = load_raw(spreadsheet)

    fake_ws.update.assert_called_once()
    assert ws == fake_ws


def test_clean_data_transforms_columns():
    worksheet = MagicMock()
    worksheet.get_all_records.return_value = [
        {
            "Unnamed: 0": 0,
            "Title": "Nice",
            "Review Text": "Loved it",
            "Class Name": "Initmates",
            "Division Name": "General",
            "Department Name": "Tops",
        }
    ]

    spreadsheet = MagicMock()
    fake_ws = MagicMock()
    spreadsheet.worksheets.return_value = []
    spreadsheet.add_worksheet.return_value = fake_ws

    df = clean_data(spreadsheet, worksheet)

    assert "title" in df.columns
    assert "review_text" in df.columns
    assert df["class_name"].iloc[0] == "Intimates"  # spelling corrected



# @patch("your_module_filename.Groq")
# def test_summarization(mock_groq):
#     # Mock Groq client and response
#     mock_client_instance = MagicMock()
#     mock_groq.return_value = mock_client_instance
#     mock_response = MagicMock()
#     mock_response.choices = [
#         MagicMock(message=MagicMock(content=json.dumps([
#             {"summary": "Great product", "sentiment": "positive"}
#         ])))
#     ]
#     mock_client_instance.chat.completions.create.return_value = mock_response

#     # Sample DataFrame
#     df = pd.DataFrame({
#         "review_text": ["I loved it", "Not good", ""]
#     })

#     # Mock spreadsheet & worksheet
#     spreadsheet = MagicMock()
#     fake_ws = MagicMock()
#     spreadsheet.worksheets.return_value = []
#     spreadsheet.add_worksheet.return_value = fake_ws

#     from your_module_filename import summarization
#     result_df = summarization(df, spreadsheet, review_column="review_text")

#     # Check columns created
#     assert "summary" in result_df.columns
#     assert "sentiment" in result_df.columns
#     assert "action_needed" in result_df.columns

#     # Check that empty review is handled
#     assert result_df["summary"].iloc[2] is None

#     # Check values from mock API
#     assert result_df["summary"].iloc[0] == "Great product"
#     assert result_df["sentiment"].iloc[0] == "positive"

#     # Ensure sheet update was called
#     spreadsheet.add_worksheet.assert_called()
#     fake_ws.update.assert_called()

