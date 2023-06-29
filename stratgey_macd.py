# -*- coding: utf-8 -*-
# @Time : 2023/6/29 16:20
# @Author : zihua.zeng
# @File : macd_strategy.py

import backtrader as bt

class MACDStrategy(bt.Strategy):
    """

    """

    params = (("profile", 0.1),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.macd = bt.indicators.MACD(self.dataclose)
        self.macdhist = bt.indicators.MACDHisto(self.dataclose)

        # 跟踪订单，同一时间只有一个订单
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def next(self):
        self.log(f"Close, {self.dataclose[0]}")

        if self.order:
            return  # 如果有订单，就不执行下面的操作

        if not self.position:
            if (self.macd[-1] < self.macd.signal[-1]) and (self.macd[0] > self.macd.signal[0]):
                self.order = self.buy()

        else:
            stop_profile_price = self.position.price * (1.0 + self.params.profile)
            stop_loss_price = self.position.price * (1.0 - self.params.profile)

            if self.dataclose[0] > stop_profile_price:
                self.order = self.sell()
                return

            if self.dataclose[0] < stop_loss_price:
                self.order = self.sell()
                return

            if len(self) >= (self.bar_executed + 3):
                if self.macd[0] < self.macd.signal[0]:
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

            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f' %
                         (order.executed.price))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f, VALUE %.2f' %
                 (trade.pnl, trade.pnlcomm, self.broker.getvalue()))

    def stop(self):
        self.log('(Threshold %.4f) Ending Value %.2f' %
                 (self.params.profile, self.broker.getvalue()))
