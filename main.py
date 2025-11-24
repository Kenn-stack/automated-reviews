from src.utils import authenticate, check_idempotence
from src.etl import run_etl

def main():
    spreadsheet = authenticate()
    start_pipeline = check_idempotence(spreadsheet)
    if start_pipeline:
        result = run_etl(spreadsheet)
        return result



if __name__ == "__main__":
    main()
