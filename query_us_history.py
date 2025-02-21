import yfinance as yf
import pandas as pd
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

def get_us_stock_history(symbol, start_date, end_date, interval='1d', save_path=None, plot=True):
    """
    获取美股历史K线数据，保存为CSV并绘制走势图
    
    参数:
        symbol (str): 股票代码 (例如: 'AAPL', 'GOOGL')
        start_date (str): 开始日期 (格式: 'YYYY-MM-DD')
        end_date (str): 结束日期 (格式: 'YYYY-MM-DD')
        interval (str): K线间隔 ('1d'=日线, '1wk'=周线, '1mo'=月线, '1m'=1分钟, '5m'=5分钟)
        save_path (str): CSV保存路径，默认为None (将保存在当前目录)
        plot (bool): 是否生成走势图，默认为True
    
    返回:
        pandas.DataFrame: 历史数据DataFrame
    """
    try:
        # 创建股票对象
        stock = yf.Ticker(symbol)
        
        # 获取历史数据
        df = stock.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        # 如果数据为空，抛出异常
        if df.empty:
            raise ValueError(f"未能获取到 {symbol} 的数据")
        
        # 重置索引，将日期变成一列
        df.reset_index(inplace=True)
        # 去除时区信息并转换为YYYY-MM-DD格式
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        
        # 如果未指定保存路径，使用默认命名
        if save_path is None:
            save_path = f"{symbol}_{start_date}_{end_date}_{interval}.csv"
        
        # 保存为CSV文件
        df.to_csv(save_path, index=False)
        print(f"数据已保存到: {save_path}")
        
        # 如果plot为True，才绘制和保存走势图
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date'], df['Close'], label='Close Price')
            plt.title(f'{symbol} Stock Price ({start_date} to {end_date})')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            plot_path = save_path.replace('.csv', '.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"走势图已保存到: {plot_path}")
        
        return df
        
    except Exception as e:
        print(f"获取数据或绘制图表时发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='获取美股历史数据')
    parser.add_argument('--symbol', type=str, required=True, help='股票代码 (例如: AAPL, GOOGL)')
    parser.add_argument('--start_date', type=str, required=True, help='开始日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='结束日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d', help='K线间隔 (1d=日线, 1wk=周线, 1mo=月线, 默认: 1d)')
    parser.add_argument('--no-plot', action='store_true', help='不生成走势图')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取数据
    df = get_us_stock_history(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        plot=not args.no_plot
    )
