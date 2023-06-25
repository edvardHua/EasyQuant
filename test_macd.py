# -*- coding: utf-8 -*-
# @Time : 2023/6/25 16:06
# @Author : zihua.zeng
# @File : test.py

import talib as ta
import pandas as pd
import matplotlib.pyplot as plt

stock_df = pd.read_csv("assets/hk700_snapshot.csv")

# 下面指标的计算使用收盘价格
dif, dea, hist = ta.MACD(stock_df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
ema12 = ta.EMA(stock_df["close"], timeperiod=12)
ema26 = ta.EMA(stock_df["close"], timeperiod=26)

stock_df_part = stock_df.loc[:, ["time_key", "close", "open"]]

# 计算买入信号
stock_df_part['sig1'] = hist > 0
stock_df_part['sig2'] = (hist > 0) & (dea > 0)
stock_df_part['sig3'] = (hist > 0) & (stock_df['close'] > ema26)
# 转换 bool 为 int
stock_df_part['sig1'] = stock_df_part['sig1'].astype(int)
stock_df_part['sig2'] = stock_df_part['sig2'].astype(int)
stock_df_part['sig3'] = stock_df_part['sig3'].astype(int)

fig, ax = plt.subplots(4, 1, figsize=(18, 12))

stock_df_part.plot(x="time_key", y="close", ax=ax[0])
stock_df_part.plot(x='time_key', y='sig1', ax=ax[1])
stock_df_part.plot(x='time_key', y='sig2', ax=ax[2])
stock_df_part.plot(x='time_key', y='sig3', ax=ax[3])
plt.show()
