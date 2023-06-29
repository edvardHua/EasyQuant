# -*- coding: utf-8 -*-
# @Time : 2023/6/29 09:33
# @Author : zihua.zeng
# @File : data_source.py

import yfinance


def obtain_us_stock_data(code, start_date, end_date):
    """
    返回的 DataFrame 中，column name
    Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')

    :param code: str
    :param start_date: str, 2016-01-01
    :param end_date: str, 2022-01-01
    :return: pd.DataFrame
    """
    try:
        data = yfinance.download(code, start=start_date, end=end_date)
    except Exception as e:
        print("[ERROR] - US stock data download failed, ", e)
        exit(0)
    print("[INFO] - Data downloaded.")
    return data


def obtain_hk_stock_data():
    pass


def obtain_cn_stock_data():
    pass


if __name__ == '__main__':
    pass
