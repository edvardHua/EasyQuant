#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data_with_date_range
# 定义低买高抛策略
class LowBuyHighSellStrategy(bt.Strategy):
    params = (
        ('rsi_period_6', 6),  # RSI周期
        ('rsi_period', 14),  # RSI周期
        ('rsi_overbought', 70),  # RSI超买阈值
        ('rsi_oversold', 30),  # RSI超卖阈值
        ('ma_period', 20),  # 移动平均线周期
        ('printlog', False),  # 是否打印日志
    )

    def log(self, txt, dt=None, doprint=False):
        """记录策略信息"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        # 保存引用到收盘价
        self.dataclose = self.datas[0].close
        
        # 跟踪订单
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # 添加指标
        self.rsi_6 = bt.indicators.RelativeStrengthIndex(
            self.datas[0], period=self.params.rsi_period_6)
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.datas[0], period=self.params.rsi_period)
        self.ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_period)

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交/接受 - 无操作
            return

        # 检查订单是否已完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'买入执行, 价格: {order.executed.price:.2f}, '
                    f'成本: {order.executed.value:.2f}, '
                    f'手续费: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # 卖出
                self.log(
                    f'卖出执行, 价格: {order.executed.price:.2f}, '
                    f'成本: {order.executed.value:.2f}, '
                    f'手续费: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')

        # 重置订单
        self.order = None

    def notify_trade(self, trade):
        """交易状态通知"""
        if not trade.isclosed:
            return

        self.log(f'交易利润, 毛利润: {trade.pnl:.2f}, 净利润: {trade.pnlcomm:.2f}')

    def next(self):
        """下一个交易日"""
        # 记录收盘价
        self.log(f'收盘价, {self.dataclose[0]:.2f}, RSI_6: {self.rsi_6[0]:.2f}, RSI: {self.rsi[0]:.2f}, MA: {self.ma[0]:.2f}')

        # 如果有未完成的订单，不操作
        if self.order:
            return

        # 检查是否持仓
        if not self.position:
            # 如果没有持仓，检查是否满足买入条件
            # 条件1: RSI低于超卖阈值
            # 条件2: 收盘价低于移动平均线
            if self.rsi[0] < self.params.rsi_oversold and self.dataclose[0] < self.ma[0]:
                self.log(f'买入信号, RSI: {self.rsi[0]:.2f}, 收盘价: {self.dataclose[0]:.2f}, MA: {self.ma[0]:.2f}')
                # 买入
                self.order = self.buy()
        else:
            # 如果已持仓，检查是否满足卖出条件
            # 条件1: RSI高于超买阈值
            # 条件2: 收盘价高于移动平均线
            if self.rsi[0] > self.params.rsi_overbought and self.dataclose[0] > self.ma[0]:
                self.log(f'卖出信号, RSI: {self.rsi[0]:.2f}, 收盘价: {self.dataclose[0]:.2f}, MA: {self.ma[0]:.2f}')
                # 卖出
                self.order = self.sell()

    def stop(self):
        """策略结束时调用"""
        self.log('(RSI: %2d) 期末资产值: %.2f' %
                 (self.params.rsi_period, self.broker.getvalue()), doprint=True)



if __name__ == '__main__':
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(LowBuyHighSellStrategy, printlog=True)

    # 读取数据
    data = load_data_with_date_range(file_path='NIO_2005-01-01_2025-03-03_1d.csv', start_date='2024-12-01', end_date='2025-03-03')
    
    # 创建数据源
    data_feed = bt.feeds.PandasData(dataname=data)
    
    # 添加数据到引擎
    cerebro.adddata(data_feed)

    # 设置初始资金
    cerebro.broker.setcash(100000.0)
    
    # 设置佣金
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 设置每次交易的股票数量
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    # 打印初始资金
    print('初始资金: %.2f' % cerebro.broker.getvalue())

    # 运行回测
    cerebro.run()

    # 打印最终资金
    print('最终资金: %.2f' % cerebro.broker.getvalue())

    # 绘制结果
    cerebro.plot(style='candle', barup='red', bardown='green')
