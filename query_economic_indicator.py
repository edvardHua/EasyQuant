import pandas as pd
import pandas_datareader.data as web
import requests
from datetime import datetime
import json
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

def get_gdp():
    """获取美国GDP数据"""
    try:
        # 从FRED获取实际GDP数据
        gdp = web.DataReader('GDP', 'fred', start=datetime(2010, 1, 1))
        return gdp
    except Exception as e:
        print(f"获取GDP数据时出错: {e}")
        return None

def get_cpi():
    """获取美国CPI数据"""
    try:
        # 从FRED获取CPI数据
        cpi = web.DataReader('CPIAUCSL', 'fred', start=datetime(2010, 1, 1))
        return cpi
    except Exception as e:
        print(f"获取CPI数据时出错: {e}")
        return None

def get_pmi():
    """获取美国PMI数据"""
    try:
        # 从FRED获取制造业PMI数据
        pmi = web.DataReader('NAPM', 'fred', start=datetime(2010, 1, 1))
        return pmi
    except Exception as e:
        print(f"获取PMI数据时出错: {e}")
        return None

def get_pce():
    """获取美国PCE数据"""
    try:
        # 从FRED获取PCE数据
        pce = web.DataReader('PCE', 'fred', start=datetime(2010, 1, 1))
        return pce
    except Exception as e:
        print(f"获取PCE数据时出错: {e}")
        return None

def plot_indicators(gdp_data, cpi_data, pmi_data, pce_data):
    """绘制经济指标图表"""
    # 创建一个4行1列的子图
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))
    
    # 绘制GDP走势
    if gdp_data is not None:
        ax1.plot(gdp_data.index, gdp_data['GDP'], 'b-', label='GDP')
        ax1.set_title('GDP Trend')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('GDP (billion usd)')
        ax1.grid(True)
        ax1.legend()
    
    # 绘制CPI走势
    if cpi_data is not None:
        ax2.plot(cpi_data.index, cpi_data['CPIAUCSL'], 'r-', label='CPI')
        ax2.set_title('CPI Trend')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('CPI Index')
        ax2.grid(True)
        ax2.legend()
    
    # 绘制PMI走势
    if pmi_data is not None:
        ax3.plot(pmi_data.index, pmi_data['NAPM'], 'g-', label='PMI')
        ax3.set_title('PMI Trend')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('PMI Index')
        ax3.grid(True)
        ax3.legend()
        
        # 添加PMI荣枯线
        ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    
    # 绘制PCE走势
    if pce_data is not None:
        ax4.plot(pce_data.index, pce_data['PCE'], 'purple', label='PCE')
        ax4.set_title('PCE走势')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('PCE (十亿美元)')
        ax4.grid(True)
        ax4.legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('economic_indicators.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为 'economic_indicators.png'")

def calculate_growth_rates(data, column_name):
    """计算环比和同比增长率"""
    # 将索引转换为日期类型
    data.index = pd.to_datetime(data.index)
    
    # 计算环比增长率（与上期相比）
    mom = data[column_name].pct_change() * 100
    
    # 计算同比增长率（与去年同期相比）
    yoy = data[column_name].pct_change(periods=4) * 100  # GDP是季度数据
    if column_name == 'CPIAUCSL':
        yoy = data[column_name].pct_change(periods=12) * 100  # CPI是月度数据
    
    return pd.DataFrame({
        column_name: data[column_name],
        'MoM': mom,
        'YoY': yoy
    })

def plot_growth_rates(gdp_data, cpi_data, pce_data):
    """绘制增长率图表"""
    if gdp_data is not None:
        gdp_growth = calculate_growth_rates(gdp_data, 'GDP')
        
        # 创建GDP增长率图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(gdp_growth.index, gdp_growth['MoM'], 'b-', label='QoQ')
        ax1.set_title('GDP QoQ')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Growth Rate (%)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(gdp_growth.index, gdp_growth['YoY'], 'r-', label='YoY')
        ax2.set_title('GDP YoY')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Growth Rate (%)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('gdp_growth_rates.png', dpi=300, bbox_inches='tight')
        print("\nGDP增长率图表已保存为 'gdp_growth_rates.png'")
        gdp_growth.to_csv('gdp_growth_rates.csv')
        print("GDP增长率数据已保存为 'gdp_growth_rates.csv'")
    
    if cpi_data is not None:
        cpi_growth = calculate_growth_rates(cpi_data, 'CPIAUCSL')
        
        # 创建CPI增长率图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(cpi_growth.index, cpi_growth['MoM'], 'b-', label='QoQ')
        ax1.set_title('CPI QoQ')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Growth Rate (%)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(cpi_growth.index, cpi_growth['YoY'], 'r-', label='YoY')
        ax2.set_title('CPI YoY')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Growth Rate (%)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('cpi_growth_rates.png', dpi=300, bbox_inches='tight')
        print("\nCPI增长率图表已保存为 'cpi_growth_rates.png'")
        cpi_growth.to_csv('cpi_growth_rates.csv')
        print("CPI增长率数据已保存为 'cpi_growth_rates.csv'")
    
    if pce_data is not None:
        pce_growth = calculate_growth_rates(pce_data, 'PCE')
        
        # 创建PCE增长率图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(pce_growth.index, pce_growth['MoM'], 'b-', label='QoQ')
        ax1.set_title('PCE QoQ')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Growth Rate (%)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(pce_growth.index, pce_growth['YoY'], 'r-', label='YoY')
        ax2.set_title('PCE YoY')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Growth Rate (%)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('pce_growth_rates.png', dpi=300, bbox_inches='tight')
        print("\nPCE增长率图表已保存为 'pce_growth_rates.png'")
        pce_growth.to_csv('pce_growth_rates.csv')
        print("PCE增长率数据已保存为 'pce_growth_rates.csv'")

def main():
    # 获取数据
    gdp_data = get_gdp()
    cpi_data = get_cpi()
    pmi_data = get_pmi()
    pce_data = get_pce()
    
    # 打印最新数据
    if gdp_data is not None:
        print("\nGDP最新数据:")
        print(gdp_data.tail())
    
    if cpi_data is not None:
        print("\nCPI最新数据:")
        print(cpi_data.tail())
    
    if pmi_data is not None:
        print("\nPMI最新数据:")
        print(pmi_data.tail())
    
    if pce_data is not None:
        print("\nPCE最新数据:")
        print(pce_data.tail())
    
    # 绘制并保存原始数据图表
    plot_indicators(gdp_data, cpi_data, pmi_data, pce_data)
    
    # 绘制并保存增长率图表
    plot_growth_rates(gdp_data, cpi_data, pce_data)
    
    # 保存原始数据到CSV文件
    if gdp_data is not None:
        gdp_data.to_csv('gdp_data.csv')
        print("\nGDP数据已保存为 'gdp_data.csv'")
    
    if cpi_data is not None:
        cpi_data.to_csv('cpi_data.csv')
        print("\nCPI数据已保存为 'cpi_data.csv'")
    
    if pce_data is not None:
        pce_data.to_csv('pce_data.csv')
        print("\nPCE数据已保存为 'pce_data.csv'")

if __name__ == "__main__":
    main()



