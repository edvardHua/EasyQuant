import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from query_us_history import get_us_stock_history
from utils import load_data_with_date_range


def analyze_price_changes(df, window_size=7):
    """
    分析股票价格变化，计算每个交易日往前window_size个交易日的涨跌幅百分比
    并预测未来一天和未来三天的涨跌情况
    
    参数:
    df (pd.DataFrame): 包含股票数据的DataFrame
    window_size (int): 时间窗口大小，默认为7个交易日
    
    返回:
    pd.DataFrame: 包含涨跌幅数据和未来涨跌预测的DataFrame
    """
    
    # 确保数据按时间顺序排列
    
    # 创建日期索引
    if 'Date' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        # 创建一个假设的日期序列，从今天开始向前推
        dates = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='B')[::-1]
        df['Date'] = dates
    
    # 计算每个交易日往前window_size个交易日的涨跌幅
    price_changes = []
    
    for i in range(window_size - 1, len(df)):
        current_price = df.iloc[i]['Close']
        past_price = df.iloc[i - (window_size - 1)]['Close']
        
        # 获取日期
        if isinstance(df.index, pd.DatetimeIndex):
            current_date = df.index[i]
            past_date = df.index[i - (window_size - 1)]
        else:
            current_date = df.iloc[i]['Date']
            past_date = df.iloc[i - (window_size - 1)]['Date']
        
        # 计算涨跌幅百分比 (当前价格相对于过去价格的变化)
        percent_change = ((current_price - past_price) / past_price) * 100
        
        # 判断未来一天是涨还是跌
        next_day_trend = 0  # 默认为0（跌）
        if i + 1 < len(df):  # 确保有下一个交易日
            next_day_price = df.iloc[i + 1]['Close']
            if next_day_price > current_price:
                next_day_trend = 1  # 涨为1
        
        # 判断未来三天是涨还是跌
        next_three_day_trend = 0  # 默认为0（跌）
        if i + 3 < len(df):  # 确保有未来第三个交易日
            next_three_day_price = df.iloc[i + 3]['Close']
            if next_three_day_price > current_price:
                next_three_day_trend = 1  # 涨为1
        
        price_changes.append({
            'Current_Index': i,
            'Date': current_date,
            'Past_Index': i - (window_size - 1),
            'Current_Date': current_date,
            'Past_Date': past_date,
            'Current_Price': current_price,
            'Past_Price': past_price,
            'Percent_Change': percent_change,
            'Next_Day_Trend': next_day_trend,  # 添加未来一天涨跌的标记
            'Next_Three_Day_Trend': next_three_day_trend  # 添加未来三天涨跌的标记
        })
    
    # 创建DataFrame
    result_df = pd.DataFrame(price_changes)
    
    # # 保存结果到CSV
    result_df.to_csv('price_changes_analysis.csv', index=False)
    
    return result_df


def visualize_price_change_distribution(df, output_file='price_change_distribution.png'):
    """
    绘制涨跌幅百分比的分布图并保存
    
    参数:
    df (pd.DataFrame): 包含涨跌幅数据的DataFrame
    output_file (str): 输出图片文件名
    
    返回:
    None
    """
    try:
        # 设置风格
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 8))
        
        # 创建分布图
        ax = sns.histplot(df['Percent_Change'], kde=True, bins=30, color='#4287f5')
        sns.despine(left=True, bottom=True)
        
        # 添加均值和中位数的垂直线
        mean_val = df['Percent_Change'].mean()
        median_val = df['Percent_Change'].median()
        
        plt.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
                   label=f'均值: {mean_val:.2f}%')
        plt.axvline(median_val, color='#2ecc71', linestyle='-.', linewidth=2, 
                   label=f'中位数: {median_val:.2f}%')
        
        # 计算百分位数
        percentiles = [10, 25, 75, 90]
        percentile_vals = np.percentile(df['Percent_Change'], percentiles)
        
        # 添加百分位区域的阴影
        for i, p in enumerate(percentiles):
            if i == 0:  # 10% 百分位
                plt.axvline(percentile_vals[i], color='#9b59b6', linestyle=':', linewidth=1.5,
                           label=f'{p}% 百分位: {percentile_vals[i]:.2f}%')
            elif i == len(percentiles) - 1:  # 90% 百分位
                plt.axvline(percentile_vals[i], color='#9b59b6', linestyle=':', linewidth=1.5,
                           label=f'{p}% 百分位: {percentile_vals[i]:.2f}%')
        
        # 添加标题和标签
        plt.title('股票价格变化百分比分布', fontsize=18, fontweight='bold')
        plt.xlabel('价格变化百分比 (%)', fontsize=14)
        plt.ylabel('频率', fontsize=14)
        
        # 添加统计信息文本框
        stats_text = (f"样本数量: {len(df)}\n"
                      f"均值: {mean_val:.2f}%\n"
                      f"中位数: {median_val:.2f}%\n"
                      f"标准差: {df['Percent_Change'].std():.2f}%\n"
                      f"最小值: {df['Percent_Change'].min():.2f}%\n"
                      f"最大值: {df['Percent_Change'].max():.2f}%")
        
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                    va='top', fontsize=12)
        
        # 添加图例
        plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        
        # 优化布局并保存图片
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"分布图已保存至 {output_file}")
        
        # 显示图片
        plt.show()
        
    except ImportError as e:
        print(f"无法创建图表：{e}")
        print("请安装必要的库：pip install matplotlib seaborn")


# 执行分析
if __name__ == "__main__":
    df = get_us_stock_history(symbol='QQQ', start_date='2015-01-01', end_date='2025-02-13', interval='1d', save_path=None, plot=False)
    
    price_changes_df = analyze_price_changes(df, window_size=7)
    
    # 绘制并保存涨跌幅分布图
    visualize_price_change_distribution(price_changes_df)





