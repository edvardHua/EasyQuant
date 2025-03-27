import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from scipy import stats

from query_us_history import get_us_stock_history
from strategy_stock_statistic import analyze_price_changes

# 设置页面配置
st.set_page_config(
    page_title="股票涨跌幅监控面板",
    page_icon="📈",
    layout="wide"
)

# 应用标题
st.title("📊 股票涨跌幅监控分析面板")
st.markdown("---")

# 侧边栏 - 参数设置
with st.sidebar:
    st.header("参数设置")
    
    # 股票标的选择
    default_symbols = ["QQQ", "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD"]
    selected_symbol = st.selectbox(
        "选择股票标的", 
        options=default_symbols,
        index=0
    )
    
    # 自定义股票标的
    custom_symbol = st.text_input("或输入其他股票代码")
    if custom_symbol:
        selected_symbol = custom_symbol
    
    # 日期区间选择
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now() - timedelta(days=1)
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # 时间窗口选择
    window_sizes = [3, 7, 14]
    selected_windows = st.multiselect(
        "选择分析窗口大小 (天)",
        options=window_sizes,
        default=window_sizes
    )
    
    # 自定义窗口大小
    custom_window = st.number_input("自定义窗口大小", min_value=1, max_value=60, value=1)
    if custom_window > 0 and custom_window not in selected_windows:
        selected_windows.append(int(custom_window))

    # 刷新按钮
    refresh = st.button("刷新数据", use_container_width=True)

# 主要函数
def calculate_technical_indicators(df):
    """计算技术指标"""
    # 计算 EMA 均线
    df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA60'] = ta.trend.ema_indicator(df['Close'], window=60)
    
    # 计算 MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    return df

def plot_candlestick(df):
    """绘制蜡烛图和技术指标"""
    # 复制一份数据，以防需要修改
    df = df.copy()
    
    # 确保索引是日期类型
    if not isinstance(df.index, pd.DatetimeIndex):
        # 尝试将索引转换为日期类型
        try:
            if 'Date' in df.columns:
                # 如果有Date列，使用它作为索引
                df = df.set_index('Date')
            else:
                # 创建一个临时日期索引
                df['_temp_date'] = pd.date_range(start='2000-01-01', periods=len(df))
                df = df.set_index('_temp_date')
        except Exception as e:
            st.warning(f"无法将索引转换为日期格式: {e}")
    
    # 创建子图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3])
    
    # 蜡烛图 - 修复悬停提示
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="蜡烛图",
            hoverinfo='none'  # 禁用默认悬停信息
        ),
        row=1, col=1
    )
    
    # 添加隐藏的散点图用于显示自定义悬停信息
    hover_text = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        # 检查索引类型，处理非日期索引的情况
        if isinstance(idx, (datetime, pd.Timestamp)):
            date_str = idx.strftime('%Y-%m-%d')
        else:
            # 如果索引不是日期类型，尝试使用Date列或创建一个通用标签
            if 'Date' in row:
                date_str = row['Date']
                if isinstance(date_str, (datetime, pd.Timestamp)):
                    date_str = date_str.strftime('%Y-%m-%d')
            else:
                date_str = f"数据点 {i+1}"
        
        # 创建悬停文本
        text = (
            f"<b>{date_str}</b><br>" +
            f"开盘: {row['Open']:.2f}<br>" +
            f"最高: {row['High']:.2f}<br>" +
            f"最低: {row['Low']:.2f}<br>" +
            f"收盘: {row['Close']:.2f}<br>" +
            f"成交量: {row['Volume']:,.0f}"
        )
        hover_text.append(text)
    
    # 添加隐藏的散点图来显示蜡烛图的悬停信息
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='markers',
            marker=dict(size=0, opacity=0),
            showlegend=False,
            hoverinfo='text',
            hovertext=hover_text
        ),
        row=1, col=1
    )
    
    # 添加均线 - 添加详细的悬停提示
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA5'],
            name="EMA5",
            line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5),
            hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>" +
            "EMA5: %{y:.2f}<br>" +
            "<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA20'],
            name="EMA20",
            line=dict(color='rgba(255, 0, 0, 0.8)', width=1.5),
            hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>" +
            "EMA20: %{y:.2f}<br>" +
            "<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA60'],
            name="EMA60",
            line=dict(color='rgba(0, 0, 255, 0.8)', width=1.5),
            hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>" +
            "EMA60: %{y:.2f}<br>" +
            "<extra></extra>"
        ),
        row=1, col=1
    )
    
    # 添加 MACD - 添加详细的悬停提示
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD'],
            name="MACD",
            line=dict(color='rgba(0, 0, 255, 0.8)', width=1),
            hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>" +
            "MACD: %{y:.4f}<br>" +
            "<extra></extra>"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            name="Signal",
            line=dict(color='rgba(255, 0, 0, 0.8)', width=1),
            hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>" +
            "Signal: %{y:.4f}<br>" +
            "<extra></extra>"
        ),
        row=2, col=1
    )
    
    # MACD 柱状图 - 添加详细的悬停提示
    colors = ['red' if x < 0 else 'green' for x in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['MACD_Hist'],
            name="MACD Histogram",
            marker_color=colors,
            hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>" +
            "Histogram: %{y:.4f}<br>" +
            "<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 更新布局，设置悬停模式为"x unified"以在同一x位置显示所有数据
    fig.update_layout(
        title=f'{selected_symbol} 价格走势 ({start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")})',
        xaxis_title="日期",
        yaxis_title="价格",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified"
    )
    
    return fig

def plot_percent_change(df_list, window_sizes):
    """绘制涨跌百分比走势图"""
    fig = go.Figure()
    
    colors = ['rgba(0, 128, 255, 0.8)', 'rgba(255, 0, 0, 0.8)', 'rgba(0, 255, 0, 0.8)', 'rgba(128, 0, 128, 0.8)']
    
    for i, (df, window) in enumerate(zip(df_list, window_sizes)):
        # 确保日期列是日期类型
        date_col = df['Date']
        if not pd.api.types.is_datetime64_any_dtype(date_col):
            try:
                # 尝试转换为日期类型
                date_col = pd.to_datetime(date_col)
            except:
                # 如果无法转换，保持原样
                pass
        
        fig.add_trace(
            go.Scatter(
                x=date_col,
                y=df['Percent_Change'],
                name=f"{window}天涨跌幅",
                line=dict(color=colors[i % len(colors)], width=1.5)
            )
        )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # 更新布局
    fig.update_layout(
        title="不同时间窗口的涨跌幅变化",
        xaxis_title="日期",
        yaxis_title="涨跌幅 (%)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def create_distribution_plot(df, window_size):
    """创建涨跌幅分布图"""
    # 获取数据
    mean_val = df['Percent_Change'].mean()
    median_val = df['Percent_Change'].median()
    latest_value = df.iloc[-1]['Percent_Change']
    
    # 使用Plotly创建直方图
    fig = go.Figure()
    
    # 添加直方图与KDE曲线
    fig.add_trace(go.Histogram(
        x=df['Percent_Change'],
        name='频率',
        opacity=0.7,
        marker_color='#4287f5',
        nbinsx=30,
        histnorm='probability density'
    ))
    
    # 添加KDE曲线 - 使用scipy.stats计算KDE并处理可能的错误
    try:
        # 检查数据是否有足够的变化
        if df['Percent_Change'].nunique() > 3:  # 确保至少有几个不同的值
            # 添加微小的噪声避免奇异矩阵
            percent_change_with_noise = df['Percent_Change'] + np.random.normal(0, 1e-6, len(df))
            kde = stats.gaussian_kde(percent_change_with_noise)
            kde_x = np.linspace(df['Percent_Change'].min(), df['Percent_Change'].max(), 100)
            kde_y = kde(kde_x)
            
            # 添加KDE曲线
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='密度分布',
                line=dict(color='#4287f5', width=2)
            ))
    except Exception as e:
        st.warning(f"无法生成KDE曲线: {e}，仅显示直方图")
    
    # 添加均值线
    fig.add_vline(x=mean_val, line_dash='dash', line_color='#e74c3c',
                  annotation_text=f'均值: {mean_val:.2f}%', 
                  annotation_position="top right")
    
    # 添加中位数线
    fig.add_vline(x=median_val, line_dash='dashdot', line_color='#2ecc71',
                  annotation_text=f'中位数: {median_val:.2f}%', 
                  annotation_position="top left")
    
    # 添加当前值线
    fig.add_vline(x=latest_value, line_color='purple',
                  annotation_text=f'当前值: {latest_value:.2f}%', 
                  annotation_position="bottom right")
    
    # 更新布局
    fig.update_layout(
        title=f'{window_size}天窗口涨跌幅分布',
        xaxis_title='涨跌幅 (%)',
        yaxis_title='频率',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def get_statistics(df_list, window_sizes):
    """计算并返回统计信息"""
    stats = []
    
    for df, window in zip(df_list, window_sizes):
        # 获取最新的涨跌幅
        latest_value = df.iloc[-1]['Percent_Change']
        
        # 计算百分位
        percentile = stats_percentile(df['Percent_Change'].values, latest_value)
        
        # 计算占比 (处于相似区间的占总样本的比例)
        similar_range = 0.5  # 定义相似区间范围为±0.5%
        similar_count = len(df[(df['Percent_Change'] >= latest_value - similar_range) & 
                              (df['Percent_Change'] <= latest_value + similar_range)])
        proportion = (similar_count / len(df)) * 100
        
        # 计算历史上当出现类似涨跌幅后，下一个交易日涨/跌的概率
        similar_df = df[(df['Percent_Change'] >= latest_value - similar_range) & 
                      (df['Percent_Change'] <= latest_value + similar_range)]
        
        if len(similar_df) > 0:
            next_day_up_prob = (similar_df['Next_Day_Trend'].sum() / len(similar_df)) * 100
            next_day_down_prob = 100 - next_day_up_prob
            
            # 计算未来三天涨跌概率
            next_three_day_up_prob = (similar_df['Next_Three_Day_Trend'].sum() / len(similar_df)) * 100
            next_three_day_down_prob = 100 - next_three_day_up_prob
        else:
            next_day_up_prob = 0
            next_day_down_prob = 0
            next_three_day_up_prob = 0
            next_three_day_down_prob = 0
        
        stats.append({
            "window": window,
            "latest_value": latest_value,
            "percentile": percentile,
            "proportion": proportion,
            "next_day_up_prob": next_day_up_prob,
            "next_day_down_prob": next_day_down_prob,
            "next_three_day_up_prob": next_three_day_up_prob,
            "next_three_day_down_prob": next_three_day_down_prob
        })
    
    return stats

def stats_percentile(values, current_value):
    """计算当前值在序列中的百分位"""
    return sum(1 for x in values if x < current_value) / len(values) * 100

# 主程序
try:
    # 获取股票数据
    stock_data = get_us_stock_history(
        symbol=selected_symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d',
        save_path=None,
        plot=False
    )
    
    # 检查是否成功获取了数据
    if stock_data is None or len(stock_data) < 20:  # 确保有足够的数据点
        st.error(f"未能获取足够的{selected_symbol}股票数据，请检查股票代码或调整日期范围。")
        st.stop()
        
    # 计算技术指标
    stock_data_with_indicators = calculate_technical_indicators(stock_data.copy())
    
    # 绘制蜡烛图和技术指标
    fig_candlestick = plot_candlestick(stock_data_with_indicators)
    st.plotly_chart(fig_candlestick, use_container_width=True)
    
    # 分析不同窗口大小的涨跌幅
    price_changes_dfs = []
    valid_windows = []
    
    for window in selected_windows:
        # 检查窗口大小是否合适
        if window >= len(stock_data):
            st.warning(f"窗口大小({window}天)超过了可用数据长度，已跳过此窗口分析")
            continue
            
        try:
            df = analyze_price_changes(stock_data, window_size=window)
            # 检查结果是否为空
            if df is None or len(df) == 0:
                st.warning(f"无法计算{window}天窗口的涨跌幅数据，已跳过此窗口")
                continue
                
            price_changes_dfs.append(df)
            valid_windows.append(window)
        except Exception as e:
            st.warning(f"计算{window}天窗口时出错: {e}")
            continue
    
    # 检查是否有有效的窗口数据
    if len(price_changes_dfs) == 0:
        st.error("所有窗口大小都无法计算有效的涨跌幅数据，请尝试调整窗口大小或日期范围。")
        st.stop()
    
    # 更新选中的窗口大小为有效的窗口
    selected_windows = valid_windows
    
    # 绘制涨跌幅走势图
    fig_percent_change = plot_percent_change(price_changes_dfs, selected_windows)
    st.plotly_chart(fig_percent_change, use_container_width=True)
    
    # 计算统计数据
    statistics = get_statistics(price_changes_dfs, selected_windows)
    
    # 展示统计信息
    st.subheader("📊 最新统计数据分析")
    
    # 创建统计信息表格
    stats_data = []
    for stat in statistics:
        stats_data.append({
            "窗口大小 (天)": stat["window"],
            "最新涨跌幅 (%)": f"{stat['latest_value']:.2f}",
            "历史百分位 (%)": f"{stat['percentile']:.2f}",
            "相似区间占比 (%)": f"{stat['proportion']:.2f}",
            "下一交易日上涨概率 (%)": f"{stat['next_day_up_prob']:.2f}",
            "下一交易日下跌概率 (%)": f"{stat['next_day_down_prob']:.2f}",
            "未来三天上涨概率 (%)": f"{stat['next_three_day_up_prob']:.2f}",
            "未来三天下跌概率 (%)": f"{stat['next_three_day_down_prob']:.2f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df)
    
    # 使用列布局展示分布图
    st.subheader("📊 涨跌幅分布分析")
    
    # 创建列布局
    if len(selected_windows) > 0:
        cols = st.columns(min(len(selected_windows), 4))  # 每行最多4个图表
        
        for i, (window, df) in enumerate(zip(selected_windows, price_changes_dfs)):
            col_idx = i % 4  # 计算列索引
            with cols[col_idx]:
                try:
                    fig = create_distribution_plot(df, window)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"无法创建{window}天窗口的分布图: {e}")

except Exception as e:
    st.error(f"获取数据时出错: {e}")
    st.info("请检查股票代码是否正确，以及日期范围是否有效。") 