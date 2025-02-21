import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class TLTStrategy:
    def __init__(self, total_investment=120000, monthly_max_investment=10000):
        """
        初始化策略
        :param total_investment: 总投资额度
        :param monthly_max_investment: 每月最大投资额
        """
        self.total_investment = total_investment
        self.monthly_max_investment = monthly_max_investment
        self.total_invested = 0
        self.unused_cash = total_investment
        self.positions = 0
        self.current_month = None
        self.month_investment = 0

    def get_buy_strength(self, row):
        """
        返回买入信号强度和建议使用的资金比例
        """
        if pd.isna(row['RSI']) or pd.isna(row['MA20']) or pd.isna(row['MA60']):
            return 0, 0
            
        strength = 0
        ratio = 0
        
        # RSI策略
        if row['RSI'] < 30:
            strength = 3
            ratio = 1.0
        elif row['RSI'] < 35:
            strength = 2
            ratio = 0.8        
        elif row['RSI'] < 40:
            strength = 1
            ratio = 0.5
            
        # 价格显著低于MA20
        if row['Price_MA20_Ratio'] < -5:
            strength = max(strength, 2)
            ratio = max(ratio, 0.8)
        elif row['Price_MA20_Ratio'] < -3:
            strength = max(strength, 1)
            ratio = max(ratio, 0.5)
            
        # 布林带策略
        # if row['Close'] < row['BB_lower']:
        #     strength = max(strength, 2)
        #     ratio = max(ratio, 0.5)
            
        # MACD金叉
        if row['MACD'] > row['Signal'] and row['MACD'] > 0:
            strength = max(strength, 2)
            ratio = max(ratio, 0.5)
            
        return strength, ratio

    def get_total_shares(self):
        return self.positions

    def get_position_value(self, current_price):
        return self.get_total_shares() * current_price

    def log_trade(self, date, action, shares, price, amount, reason=''):
        """记录交易"""
        total_shares_after = self.get_total_shares()
        position_value = self.get_position_value(price)
        total_value = self.unused_cash + position_value
        
        return {
            'Date': date,
            'Action': action,
            'Shares': shares,
            'Price': price,
            'Amount': amount,
            'Reason': reason,
            'Monthly_Budget_Left': self.monthly_max_investment - self.month_investment,
            'Total_Shares': total_shares_after,
            'Position_Value': position_value,
            'Avg_Cost': self.total_invested / total_shares_after if total_shares_after > 0 else 0,
            'Total_Invested': self.total_invested,
            'Total_Value': total_value
        }

    def get_sell_strength(self, row):
        """
        返回卖出信号强度和建议卖出的仓位比例
        :return: (strength, ratio, reasons)
            strength: float, 卖出信号强度
            ratio: float, 建议卖出的仓位比例
            reasons: list, 卖出原因列表
        """
        if pd.isna(row['RSI']) or pd.isna(row['MA20']):
            return 0, 0, []
            
        sell_strength = 0
        sell_reason = []
        
        # RSI超买
        if row['RSI'] > 70:
            strength = min((row['RSI'] - 70) / 10, 3)  # 最大贡献为3
            sell_strength += strength
            sell_reason.append(f'RSI超买({row["RSI"]:.1f})')
        
        # 价格显著高于MA20
        if row['Price_MA20_Ratio'] > 5:
            strength = min((row['Price_MA20_Ratio'] - 5) / 2, 2)  # 最大贡献为2
            sell_strength += strength
            sell_reason.append(f'价格高于MA20({row["Price_MA20_Ratio"]:.1f}%)')
        
        # 突破布林带上轨
        if row['Close'] > row['BB_upper'] * 1.5:
            strength = min((row['Close'] / row['BB_upper'] - 1.5) * 2, 2)  # 最大贡献为2
            sell_strength += strength
            sell_reason.append('显著突破布林带上轨')
        
        # MACD死叉且为正值
        if row['MACD'] < row['Signal'] and row['MACD'] > 0:
            strength = min(row['MACD'] / 2, 1)  # 最大贡献为1
            sell_strength += strength
            sell_reason.append('MACD死叉')
            
        # 根据信号强度决定卖出比例
        if sell_strength >= 1:  # 至少要有一定的卖出信号强度
            sell_ratio = min(sell_strength / 8, 1)  # 最大卖出比例为100%
        else:
            sell_ratio = 0
        
        sell_ratio = 0
        
        return sell_strength, sell_ratio, sell_reason

    def calculate_indicators(self, df):
        """
        计算技术指标
        """
        # 计算移动平均线
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 计算布林带
        df['BB_middle'] = df['MA20']
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # 计算RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算价格相对MA20的偏离率
        df['Price_MA20_Ratio'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        
        # 计算MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        return df

    def backtest(self, df, start_date=None, end_date=None):
        """
        回测策略
        """
        # 计算技术指标
        df = self.calculate_indicators(df)
        
        # 处理日期范围
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
            
        if len(df) == 0:
            raise ValueError("No data available for the specified date range")
            
        # 记录交易
        trades = []
        
        for index, row in df.iterrows():
            current_date = pd.to_datetime(index)
            
            # 检查是否是新的月份
            if self.current_month != current_date.month:
                self.current_month = current_date.month
                self.month_investment = 0
            
            # 如果已经卖出，则当天不再买入
            mutex_flag = False
            
            # 卖出信号
            if self.positions > 0:  # 有持仓才检查卖出信号
                sell_strength, sell_ratio, sell_reason = self.get_sell_strength(row)
                
                if sell_ratio > 0:
                    shares_to_sell = int(self.positions * sell_ratio)
                    
                    if shares_to_sell > 0:
                        sell_amount = shares_to_sell * row['Close']
                        self.positions -= shares_to_sell
                        self.unused_cash += sell_amount
                        trades.append(self.log_trade(
                            index,
                            'Sell',
                            shares_to_sell,
                            row['Close'],
                            sell_amount,
                            f'信号强度:{sell_strength:.1f},' + ','.join(sell_reason)
                        ))
                        mutex_flag = True
            
            # 买入信号
            strength, ratio = self.get_buy_strength(row)
            if strength > 0 and not mutex_flag:
                available_budget = min(
                    self.monthly_max_investment - self.month_investment,
                    self.unused_cash
                )
                
                if available_budget > 0:
                    buy_amount = available_budget * ratio
                    if buy_amount >= 100:  # 最小交易金额
                        shares_to_buy = int(buy_amount / row['Close'])
                        actual_amount = shares_to_buy * row['Close']
                        
                        if actual_amount <= available_budget:
                            self.positions += shares_to_buy
                            self.unused_cash -= actual_amount
                            self.total_invested += actual_amount
                            self.month_investment += actual_amount
                            
                            # 确定买入原因
                            buy_reason = [f'信号强度:{strength}']
                            if row['RSI'] < 30:
                                buy_reason.append('RSI超卖')
                            if row['Price_MA20_Ratio'] < -5:
                                buy_reason.append('价格显著低于MA20')
                            if row['Close'] < row['BB_lower']:
                                buy_reason.append('突破布林带下轨')
                            
                            trades.append(self.log_trade(
                                index,
                                'Buy',
                                shares_to_buy,
                                row['Close'],
                                actual_amount,
                                ','.join(buy_reason)
                            ))
        
        # 保存交易日志
        if trades:
            trade_df = pd.DataFrame(trades)
            # 格式化数值列，保留2位小数
            numeric_columns = ['Price', 'Amount', 'Monthly_Budget_Left', 
                             'Position_Value', 'Avg_Cost', 'Total_Invested', 'Total_Value']
            for col in numeric_columns:
                trade_df[col] = trade_df[col].round(2)
            trade_df.to_csv('trade_log.csv', index=False)
            
        final_value = self.positions * df.iloc[-1]['Close'] + self.unused_cash
        return final_value

    def plot_strategy(self, df):
        """
        绘制策略图表
        """
        plt.figure(figsize=(15, 12))  # 增加图表高度以容纳三个子图
        
        # 绘制价格和均线
        plt.subplot(3, 1, 1)  # 修改为3行1列的第1个子图
        plt.plot(df.index, df['Close'], label='Price', color='blue')
        plt.plot(df.index, df['MA20'], label='MA20', color='orange')
        plt.plot(df.index, df['MA60'], label='MA60', color='red')
        plt.plot(df.index, df['BB_upper'], label='BB Upper', color='gray', linestyle='--')
        plt.plot(df.index, df['BB_lower'], label='BB Lower', color='gray', linestyle='--')
        
        # 标记买入点和卖出点
        buy_signals = []
        sell_signals = []
        
        for idx, row in df.iterrows():
            strength, _ = self.get_buy_strength(row)
            if strength > 0:
                buy_signals.append((idx, row['Close']))
            
            sell_strength, sell_ratio, _ = self.get_sell_strength(row)
            if sell_ratio > 0:
                sell_signals.append((idx, row['Close']))
        
        if buy_signals:
            buy_x, buy_y = zip(*buy_signals)
            plt.scatter(buy_x, buy_y, marker='^', color='green', label='Buy Signal')
        
        if sell_signals:
            sell_x, sell_y = zip(*sell_signals)
            plt.scatter(sell_x, sell_y, marker='v', color='red', label='Sell Signal')
        
        plt.title('TLT Strategy - Price and Indicators')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制RSI
        plt.subplot(3, 1, 2)  # 修改为3行1列的第2个子图
        plt.plot(df.index, df['RSI'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', label='Overbought')
        plt.axhline(y=30, color='g', linestyle='--', label='Oversold')
        plt.title('RSI Indicator')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制MACD
        plt.subplot(3, 1, 3)  # 添加第3个子图
        # 绘制MACD柱状图
        positive_macd = df['MACD_Hist'].where(df['MACD_Hist'] >= 0, 0)
        negative_macd = df['MACD_Hist'].where(df['MACD_Hist'] < 0, 0)
        plt.bar(df.index, positive_macd, color='green', label='MACD Histogram +', alpha=0.7)
        plt.bar(df.index, negative_macd, color='red', label='MACD Histogram -', alpha=0.7)
        
        # 绘制MACD和信号线
        plt.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1)
        plt.plot(df.index, df['Signal'], label='Signal', color='orange', linewidth=1)
        
        # 添加零线
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        plt.title('MACD Indicator')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('assets/tlt_strategy.png')
        plt.close()

    def calculate_annual_returns(self, df, trade_log_df):
        """
        计算策略和标的每年的收益率
        :param df: 原始价格数据
        :param trade_log_df: 交易记录数据
        :return: 策略年化收益率和标的年化收益率
        """
        # 将日期列转换为datetime并处理时区
        trade_log_df['Date'] = pd.to_datetime(trade_log_df['Date'])
        
        # 按年分组计算策略收益
        strategy_returns = {}
        for year in trade_log_df['Date'].dt.year.unique():
            year_trades = trade_log_df[trade_log_df['Date'].dt.year == year]
            if not year_trades.empty:
                start_value = year_trades.iloc[0]['Total_Value']
                end_value = year_trades.iloc[-1]['Total_Value']
                if start_value > 0:
                    strategy_returns[year] = ((end_value - start_value) / start_value) * 100
                else:
                    strategy_returns[year] = 0
        
        df.index = pd.to_datetime(df.index)
        # 计算标的年化收益
        benchmark_returns = {}
        years = df.index.year.unique()
        for year in years:
            year_data = df[df.index.year == year]
            if not year_data.empty:
                start_price = year_data.iloc[0]['Close']
                end_price = year_data.iloc[-1]['Close']
                benchmark_returns[year] = ((end_price - start_price) / start_price) * 100
                
        return strategy_returns, benchmark_returns

    def plot_annual_returns(self, strategy_returns, benchmark_returns):
        """
        绘制年化收益率对比图
        """
        plt.figure(figsize=(12, 6))
        
        years = sorted(set(list(strategy_returns.keys()) + list(benchmark_returns.keys())))
        x = np.arange(len(years))
        width = 0.35
        
        strategy_values = [strategy_returns.get(year, 0) for year in years]
        benchmark_values = [benchmark_returns.get(year, 0) for year in years]
        
        plt.bar(x - width/2, strategy_values, width, label='Strategy Returns', color='blue', alpha=0.7)
        plt.bar(x + width/2, benchmark_values, width, label='TLT Returns', color='green', alpha=0.7)
        
        plt.xlabel('Year')
        plt.ylabel('Annual Return (%)')
        plt.title('Strategy vs TLT Annual Returns')
        plt.xticks(x, years, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(strategy_values):
            plt.text(i - width/2, v + (1 if v >= 0 else -1), 
                    f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top')
        for i, v in enumerate(benchmark_values):
            plt.text(i + width/2, v + (1 if v >= 0 else -1), 
                    f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('assets/tlt_annual_returns.png')
        plt.close()

if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv('JEPQ_2015-01-01_2025-02-13_1d.csv', index_col='Date', parse_dates=True)
    
    # 创建策略实例
    strategy = TLTStrategy(
        total_investment=120000,  # 总投资额度12万
        monthly_max_investment=10000  # 每月最大投资1万
    )
    
    # 指定回测时间段
    start_date = '2024-01-01'
    end_date = '2025-01-01'
    
    try:
        final_value = strategy.backtest(df, start_date=start_date, end_date=end_date)
        
        # 读取交易记录
        trade_log = pd.read_csv('trade_log.csv')
        
        # 计算年化收益
        strategy_returns, benchmark_returns = strategy.calculate_annual_returns(df, trade_log)
        
        # 打印年化收益
        print("\nAnnual Returns:")
        print("Year\tStrategy\tTLT")
        print("-" * 30)
        for year in sorted(set(list(strategy_returns.keys()) + list(benchmark_returns.keys()))):
            strategy_return = strategy_returns.get(year, 0)
            benchmark_return = benchmark_returns.get(year, 0)
            print(f"{year}\t{strategy_return:.2f}%\t{benchmark_return:.2f}%")
        
        # 绘制年化收益对比图
        strategy.plot_annual_returns(strategy_returns, benchmark_returns)
        
        # 计算收益率（使用实际投资额而不是总额度）
        roi_act = (final_value - strategy.total_investment) / strategy.total_invested * 100 if strategy.total_invested > 0 else 0
        roi = (final_value - strategy.total_investment) / strategy.total_investment * 100
        print(f"Backtest Period: {start_date} to {end_date}")
        print(f"Total Investment Capacity: ${strategy.total_investment:.2f}")
        print(f"Final Value: ${final_value:.2f}")
        print(f"Return on Investment: {roi:.2f}%")
        print(f"Actually Invested: ${strategy.total_invested:.2f}")
        print(f"Final Portfolio Value: ${final_value - strategy.total_investment:.2f}")
        print(f"Actually Return on Investment: {roi_act:.2f}%")
        
        # 读取并显示交易记录摘要
        trade_log = pd.read_csv('trade_log.csv')
        print("\nTrading Summary:")
        print(f"Total Trades: {len(trade_log)}")
        print(f"Buy Trades: {len(trade_log[trade_log['Action'] == 'Buy'])}")
        print(f"Sell Trades: {len(trade_log[trade_log['Action'] == 'Sell'])}")
        
        # 绘制策略图表
        strategy.plot_strategy(df.loc[start_date:end_date])
        
    except ValueError as e:
        print(f"Error: {e}")
