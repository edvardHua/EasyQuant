

import pandas as pd
from utils import load_data_with_date_range


def analyze_price_changes(df, window_size=7):
    """
    分析股票价格变化，计算指定时间窗口内的涨跌幅百分比
    
    参数:
    file_path (str): CSV文件路径
    window_size (int): 时间窗口大小，默认为14个交易日
    
    返回:
    pd.DataFrame: 包含涨跌幅数据的DataFrame
    """
    
    # 确保数据按时间顺序排列（假设CSV文件中的数据已按时间顺序排列）
    
    # 创建日期索引（由于示例数据中没有日期列，我们使用索引作为日期的替代）
    # 在实际应用中，如果CSV中有日期列，应该使用实际日期
    if 'Date' not in df.columns:
        # 创建一个假设的日期序列，从今天开始向前推
        dates = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='B')[::-1]
        df['Date'] = dates
    
    # 计算每个窗口的涨跌幅
    price_changes = []
    
    for i in range(len(df) - window_size + 1):
        start_price = df.iloc[i]['Close']
        end_price = df.iloc[i + window_size - 1]['Close']
        start_date = df.iloc[i]['Date']
        end_date = df.iloc[i + window_size - 1]['Date']
        
        # 计算涨跌幅百分比
        percent_change = ((end_price - start_price) / start_price) * 100
        
        price_changes.append({
            'Start_Index': i,
            'End_Index': i + window_size - 1,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Start_Price': start_price,
            'End_Price': end_price,
            'Percent_Change': percent_change
        })
    
    # 创建DataFrame
    result_df = pd.DataFrame(price_changes)
    
    # 保存结果到CSV
    result_df.to_csv('price_changes_analysis.csv', index=False)
    
    # 找出最大跌幅和最小跌幅
    max_drop = result_df['Percent_Change'].min()
    min_drop = result_df['Percent_Change'].max()
    
    # 输出结果
    print(f"分析完成! 时间窗口: {window_size}个交易日")
    print(f"最大跌幅: {max_drop:.2f}%")
    print(f"最大涨幅: {min_drop:.2f}%")
    
    return result_df

# 执行分析
if __name__ == "__main__":
    df = load_data_with_date_range(file_path='NIO_2005-01-01_2025-03-03_1d.csv', 
                                   start_date='2024-10-01', 
                                   end_date='2025-03-03')
    analyze_price_changes(df, window_size=3)
    analyze_price_changes(df, window_size=5)
    analyze_price_changes(df, window_size=7)
    analyze_price_changes(df, window_size=14)





