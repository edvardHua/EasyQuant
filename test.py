# -*- coding: utf-8 -*-
# @Time : 2023/6/29 09:37
# @Author : zihua.zeng
# @File : test.py
#

import pandas as pd
import backtrader as bt

from data_source import obtain_us_stock_data
from datetime import datetime


class MACDStrategy(bt.Strategy):
    """

    """

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.macd = bt.indicators.MACD(self.dataclose)
        self.macdhist = bt.indicators.MACDHisto(self.dataclose)

        # 跟踪订单，同一时间只有一个订单
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def next(self):
        self.log(f"Close, {self.dataclose[0]}")

        if self.order:
            return  # 如果有订单，就不执行下面的操作

        if not self.position:

            if (self.dataclose[-1] - self.dataclose[0]) > 5:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
        else:
            if len(self) >= (
                    self.bar_executed + 2):  # 这里注意，Len(self)返回的是当前执行的bar数量，每次next会加1.而Self.bar_executed记录的最后一次交易执行时的bar位置。
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))


if __name__ == '__main__':
    cerebo = bt.Cerebro()
    # cerebo.addstrategy(TestStrategy, exitbars=5)
    cerebo.addstrategy(MACDStrategy)

    data_df = obtain_us_stock_data("NIO", "2017-06-01", "2023-06-28")
    print(data_df.shape)
    start_date = datetime(2017, 6, 1)
    end_date = datetime(2023, 6, 28)

    data = bt.feeds.PandasData(dataname=data_df, fromdate=start_date, todate=end_date)
    cerebo.adddata(data)

    # 设置初始资金
    cerebo.broker.setcash(10000)
    # 设置手续费
    cerebo.broker.setcommission(0.001)

    print("Current = ", cerebo.broker.get_value())
    cerebo.run()
    print("Final = ", cerebo.broker.get_value())
