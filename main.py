# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# zhuanlan.zhihu.com/p/68375348

from futu import *

import pandas as pd

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)  # 创建行情对象
df = quote_ctx.get_market_snapshot('HK.00700')  # 获取港股 HK.00700 的快照数据

# trd_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)  # 创建交易对象
# print(trd_ctx.place_order(price=500.0, qty=100, code="HK.00700", trd_side=TrdSide.BUY, trd_env=TrdEnv.SIMULATE))  # 模拟交易，下单（如果是真实环境交易，在此之前需要先解锁交易密码）
# trd_ctx.close()  # 关闭对象，防止连接条数用尽

# 获取港股 HK.00700 (腾讯) 的 2020-01-01 到 2020-06-25 的日 K 线数据
stock = 'HK.00700'
start_date = "2023-01-01"
end_date = "2023-06-25"
ret_code, prices, page_req_key = quote_ctx.request_history_kline("HK.00700", start=start_date, end=end_date)

quote_ctx.close()  # 关闭对象，防止连接条数用尽
prices.to_csv("assets/hk700_snapshot.csv", index=False)





