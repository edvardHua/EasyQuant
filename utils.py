import pandas as pd



def load_data_with_date_range(file_path, start_date=None, end_date=None):
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index <= pd.to_datetime(end_date)]
    return data