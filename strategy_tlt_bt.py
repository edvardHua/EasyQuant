import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class PriceMAPercentage(bt.Indicator):
    """计算价格与MA的百分比差值"""
    lines = ('pct_diff',)  # 声明指标线
    params = (('period', 20),)  # 参数设置
    
    def __init__(self):
        price = self.data
        ma = bt.indicators.SimpleMovingAverage(price, period=self.p.period)
        self.lines.pct_diff = bt.indicators.PctChange(price, ma) * 100

class TLTStrategy(bt.Strategy):
    params = (
        ('total_investment', 120000),
        ('monthly_max_investment', 10000),
        ('rsi_period', 10),          # 缩短 RSI 周期以提高灵敏度
        ('bb_period', 20),
        ('atr_period', 14),          # 新增 ATR 周期参数
    )

    def __init__(self):
        # 保存交易记录
        self.trades = []
        self.current_month = None
        self.month_investment = 0
        self.total_invested = 0
        
        # 优化技术指标
        self.ma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.ma60 = bt.indicators.SimpleMovingAverage(self.data.close, period=60)
        
        # 使用更灵敏的 RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # 布林带
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.p.bb_period)
        
        # 价格与MA20的偏离度
        self.price_ma20_ratio = PriceMAPercentage(self.data.close)
        
        # MACD
        self.macd = bt.indicators.MACD(self.data.close, period_me1=12, period_me2=26, period_signal=9)
        
        # 新增 ATR 指标
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # 新增趋势强度指标
        self.adx = bt.indicators.DirectionalMovement(self.data, period=14)
        
    def get_buy_strength(self):
        """返回买入信号强度和建议使用的资金比例"""
        strength = 0
        ratio = 0
        
        # RSI 策略优化
        if self.rsi[0] < 25:  # 更激进的超卖判断
            strength = 3
            ratio = 1.0
        elif self.rsi[0] < 30:
            strength = 2
            ratio = 0.7
        elif self.rsi[0] < 35:
            strength = 1
            ratio = 0.4
            
        # 价格显著低于MA20
        ma_diff = self.price_ma20_ratio.pct_diff[0]
        if ma_diff < -7:  # 更激进的偏离判断
            strength = max(strength, 3)
            ratio = max(ratio, 0.8)
        elif ma_diff < -5:
            strength = max(strength, 2)
            ratio = max(ratio, 0.6)
            
        # 布林带策略优化
        bb_pos = (self.data.close[0] - self.bb.lines.bot[0]) / self.atr[0]
        if bb_pos < -2:  # 相对 ATR 的布林带突破
            strength = max(strength, 3)
            ratio = max(ratio, 0.8)
            
        # MACD 金叉且趋势强
        if (self.macd.macd[0] > self.macd.signal[0] and 
            self.macd.macd[-1] <= self.macd.signal[-1]):
            if self.adx.lines.adx[0] > 25:  # 强趋势
                strength = max(strength, 2)
                ratio = max(ratio, 0.6)
                
        # 考虑趋势强度
        if self.adx.lines.adx[0] > 30 and self.ma20[0] > self.ma60[0]:
            strength += 1
            ratio = min(1.0, ratio + 0.2)
            
        return strength, ratio

    def should_sell(self):
        """判断是否应该卖出"""
        # RSI 超买
        if self.rsi[0] > 75:  # 更激进的超买判断
            return True
            
        # 价格显著高于MA20
        if self.price_ma20_ratio.pct_diff[0] > 7:  # 更激进的偏离判断
            return True
            
        # 布林带上轨突破
        bb_pos = (self.data.close[0] - self.bb.lines.top[0]) / self.atr[0]
        if bb_pos > 2:  # 相对 ATR 的布林带突破
            return True
            
        # MACD 死叉且位于高位
        if (self.macd.macd[0] < self.macd.signal[0] and 
            self.macd.macd[-1] >= self.macd.signal[-1] and
            self.macd.macd[0] > 0):
            return True
            
        # 趋势反转
        if (self.adx.lines.adx[0] > 30 and 
            self.ma20[0] < self.ma60[0] and 
            self.data.close[0] < self.ma20[0]):
            return True
            
        return False

    def log_trade(self, action, shares, price, amount, reason=''):
        """记录交易"""
        position_value = self.position.size * price if self.position else 0
        cash = self.broker.getcash()
        total_value = cash + position_value
        
        # 计算交易收益率（仅针对卖出交易）
        trade_roi = 0.0
        if action == 'Sell' and shares > 0:
            trade_roi = ((price - (self.total_invested / shares)) / 
                        (self.total_invested / shares)) * 100 if self.total_invested > 0 else 0.0
        
        # 计算总体收益率
        total_roi = ((total_value - self.p.total_investment) / 
                    self.p.total_investment) * 100
        
        trade_record = {
            'Date': self.data.datetime.date(),
            'Action': action,
            'Shares': shares,
            'Price': price,
            'Amount': amount,
            'Reason': reason,
            'Cash_After_Trade': cash,
            'Monthly_Budget_Left': self.p.monthly_max_investment - self.month_investment,
            'Total_Shares': self.position.size if self.position else 0,
            'Position_Value': position_value,
            'Total_Value': total_value,
            'Avg_Cost': self.total_invested / shares if shares > 0 else 0,
            'Trade_ROI': trade_roi,
            'Total_ROI': total_roi,
            'Total_Invested': self.total_invested
        }
        
        self.trades.append(trade_record)

    def next(self):
        # 检查是否是新的月份
        current_date = self.data.datetime.date()
        if self.current_month != current_date.month:
            self.current_month = current_date.month
            self.month_investment = 0
        
        # 卖出逻辑
        if self.position and self.should_sell():
            sell_amount = self.position.size * self.data.close[0]
            
            # 确定卖出原因
            sell_reason = []
            if self.rsi[0] > 75:
                sell_reason.append('RSI超买')
            if self.price_ma20_ratio.pct_diff[0] > 7:
                sell_reason.append('价格显著高于MA20')
            if (self.data.close[0] - self.bb.lines.top[0]) / self.atr[0] > 2:
                sell_reason.append('突破布林带上轨')
            if (self.macd.macd[0] < self.macd.signal[0] and 
                self.macd.macd[-1] >= self.macd.signal[-1] and
                self.macd.macd[0] > 0):
                sell_reason.append('MACD死叉')
            
            self.log_trade(
                'Sell',
                self.position.size,
                self.data.close[0],
                sell_amount,
                ','.join(sell_reason)
            )
            self.close()
            
        # 买入逻辑
        else:
            strength, ratio = self.get_buy_strength()
            if strength > 0:
                available_budget = min(
                    self.p.monthly_max_investment - self.month_investment,
                    self.broker.getcash()
                )
                
                if available_budget > 0:
                    buy_amount = available_budget * ratio
                    if buy_amount >= 100:  # 最小交易金额
                        shares_to_buy = int(buy_amount / self.data.close[0])
                        actual_amount = shares_to_buy * self.data.close[0]
                        
                        if actual_amount <= available_budget:
                            # 确定买入原因
                            buy_reason = [f'信号强度:{strength}']
                            if self.rsi[0] < 25:
                                buy_reason.append('RSI超卖')
                            if self.price_ma20_ratio.pct_diff[0] < -7:
                                buy_reason.append('价格显著低于MA20')
                            if (self.data.close[0] - self.bb.lines.bot[0]) / self.atr[0] < -2:
                                buy_reason.append('突破布林带下轨')
                            
                            self.buy(size=shares_to_buy)
                            self.total_invested += actual_amount
                            self.month_investment += actual_amount
                            
                            self.log_trade(
                                'Buy',
                                shares_to_buy,
                                self.data.close[0],
                                actual_amount,
                                ','.join(buy_reason)
                            )

    def stop(self):
        """策略结束时保存交易记录"""
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            # 格式化数值列，保留2位小数
            numeric_columns = ['Price', 'Amount', 'Cash_After_Trade', 
                             'Monthly_Budget_Left', 'Position_Value', 
                             'Total_Value', 'Avg_Cost', 'Trade_ROI', 
                             'Total_ROI', 'Total_Invested']
            for col in numeric_columns:
                trade_df[col] = trade_df[col].round(2)
            trade_df.to_csv('trade_log_bt.csv', index=False)

def run_strategy(data_file, start_date=None, end_date=None):
    """运行回测"""
    # 创建cerebro引擎
    cerebro = bt.Cerebro()
    
    # 加载数据
    df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    # 设置初始资金
    cerebro.broker.setcash(120000.0)
    
    # 添加策略
    cerebro.addstrategy(TLTStrategy)
    
    # 运行回测
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    # 计算收益率
    roi = (final_value - initial_value) / initial_value * 100
    
    # 打印结果
    print(f'Initial Portfolio Value: ${initial_value:.2f}')
    print(f'Final Portfolio Value: ${final_value:.2f}')
    print(f'Return on Investment: {roi:.2f}%')
    
    # 绘制结果
    cerebro.plot(style='candlestick')

if __name__ == '__main__':
    # 运行回测
    run_strategy(
        'TLT_2015-01-01_2025-02-13_1d.csv',
        start_date='2024-01-01',
        end_date='2025-01-01'
    )
