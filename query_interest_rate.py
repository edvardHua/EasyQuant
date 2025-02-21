import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def get_interest_rate():
    """获取美联储利率数据"""
    try:
        # 从FRED获取联邦基金利率数据
        # FEDFUNDS: 联邦基金利率
        rate = web.DataReader('FEDFUNDS', 'fred', start=datetime(1950, 1, 1))
        return rate
    except Exception as e:
        print(f"获取利率数据时出错: {e}")
        return None

def plot_interest_rate(rate_data, start_date=None, end_date=None):
    """
    绘制利率走势图
    
    参数:
        rate_data: DataFrame, 利率数据
        start_date: str or datetime, 开始日期 (格式: 'YYYY-MM-DD' 或 datetime 对象)
        end_date: str or datetime, 结束日期 (格式: 'YYYY-MM-DD' 或 datetime 对象)
    """
    if rate_data is None:
        return
    
    # 根据时间段筛选数据
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        rate_data = rate_data[rate_data.index >= start_date]
    
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        rate_data = rate_data[rate_data.index <= end_date]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(rate_data.index, rate_data['FEDFUNDS'], linewidth=2)

    # 设置图表标题和标签
    title = 'Federal Funds Rate History'
    if start_date or end_date:
        date_range = f"({start_date.strftime('%Y-%m-%d') if start_date else 'Start'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Now'})"
        title += f"\n{date_range}"
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rate (%)', fontsize=12)

    # 根据数据时间跨度调整x轴刻度间隔
    time_span = (rate_data.index[-1] - rate_data.index[0]).days / 365  # 年数
    if time_span <= 2:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 每2个月显示一个刻度
    elif time_span <= 5:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每6个月显示一个刻度
    else:
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # 每5年显示一个刻度
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每6个月显示一个刻度

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 旋转x轴标签
    plt.xticks(rotation=45)

    # 调整布局
    plt.tight_layout()

    # 生成文件名
    filename_base = 'fed_interest_rate'
    if start_date or end_date:
        date_str = f"_{start_date.strftime('%Y%m%d') if start_date else 'start'}_{end_date.strftime('%Y%m%d') if end_date else 'end'}"
        filename_base += date_str

    # 保存图表
    plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
    print(f"\n利率走势图已保存为 '{filename_base}.png'")
    
    # 保存数据到CSV
    rate_data.to_csv(f'{filename_base}.csv')
    print(f"利率数据已保存为 '{filename_base}.csv'")

def main():
    # 获取数据
    rate_data = get_interest_rate()
    
    # 打印最新数据
    if rate_data is not None:
        print("\n最新利率数据:")
        print(rate_data.tail())
    
    # 示例：绘制2008年金融危机期间的数据
    plot_interest_rate(rate_data, start_date='2015-01-01', end_date='2025-02-13')

if __name__ == "__main__":
    main()
