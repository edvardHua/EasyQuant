# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import backtrader as bt

from datetime import datetime
from data_source import obtain_us_stock_data
from stratgey_macd import MACDStrategy


def run_one_strategy():
    """
    跑一次策略
    :return:
    """

    stock_code = "NIO"  # 股票代码
    start_date_str = "2021-01-01"  # 回测开始时间
    end_date_str = "2023-06-28"  # 回测结束时间
    init_value = 100000  # 初始资金
    comm = 0.001  # 交易佣金
    stake_size = 25  # 每手股数

    cerebo = bt.Cerebro()
    cerebo.addstrategy(MACDStrategy, profile=0.1)
    cerebo.addsizer(bt.sizers.FixedSize, stake=stake_size)
    data_df = obtain_us_stock_data(stock_code, start_date_str, end_date_str)
    stint = start_date_str.split("-")
    enint = end_date_str.split("-")
    start_date = datetime(int(stint[0]), int(stint[1]), int(stint[2]))
    end_date = datetime(int(enint[0]), int(enint[1]), int(enint[2]))

    data = bt.feeds.PandasData(dataname=data_df, fromdate=start_date, todate=end_date)
    cerebo.adddata(data)

    # 设置初始资金
    cerebo.broker.setcash(init_value)
    # 设置手续费
    cerebo.broker.setcommission(comm)
    cerebo.run()
    print("Final = ", cerebo.broker.get_value())
    cerebo.plot()


def run_multiple_strategies():
    """
    增加多参数的策略，跑多次策略
    :return:
    """

    stock_code = "NIO"  # 股票代码
    start_date_str = "2021-01-01"  # 回测开始时间
    end_date_str = "2023-06-28"  # 回测结束时间
    init_value = 100000  # 初始资金
    comm = 0.001  # 交易佣金
    stake_size = 25  # 每手股数
    profiles = [0.03, 0.05, 0.07, 0.1]  # 止盈止损参数

    cerebro = bt.Cerebro()
    # 增加多参数，用于寻找合适的参数
    cerebro.optstrategy(
        MACDStrategy,
        profile=profiles)

    # 获取数据
    stock_hfq_df = obtain_us_stock_data(stock_code, start_date_str, end_date_str)
    stint = start_date_str.split("-")
    enint = end_date_str.split("-")
    start_date = datetime(int(stint[0]), int(stint[1]), int(stint[2]))
    end_date = datetime(int(enint[0]), int(enint[1]), int(enint[2]))
    data = bt.feeds.PandasData(dataname=stock_hfq_df, fromdate=start_date, todate=end_date)  # 加载数据
    cerebro.adddata(data)  # 将数据传入回测系统

    cerebro.broker.setcash(init_value)
    cerebro.broker.setcommission(commission=comm)
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake_size)
    cerebro.run()


if __name__ == '__main__':
    # run_one_strategy()
    run_multiple_strategies()
