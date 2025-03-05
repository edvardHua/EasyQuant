

import pandas as pd
from utils import load_data_with_date_range


def analyze_price_changes(df, window_size=7):
    """
    分析股票价格变化，计算每个交易日往前window_size个交易日的涨跌幅百分比
    
    参数:
    df (pd.DataFrame): 包含股票数据的DataFrame
    window_size (int): 时间窗口大小，默认为7个交易日
    
    返回:
    pd.DataFrame: 包含涨跌幅数据的DataFrame
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
        
        price_changes.append({
            'Current_Index': i,
            'Past_Index': i - (window_size - 1),
            'Current_Date': current_date,
            'Past_Date': past_date,
            'Current_Price': current_price,
            'Past_Price': past_price,
            'Percent_Change': percent_change
        })
    
    # 创建DataFrame
    result_df = pd.DataFrame(price_changes)
    
    # # 保存结果到CSV
    # result_df.to_csv('price_changes_analysis.csv', index=False)
    
    # # 找出最大跌幅和最大涨幅
    # max_drop = result_df['Percent_Change'].min()
    # max_rise = result_df['Percent_Change'].max()
    
    # # 输出结果
    # print(f"分析完成! 时间窗口: {window_size}个交易日")
    # print(f"最大跌幅: {max_drop:.2f}%")
    # print(f"最大涨幅: {max_rise:.2f}%")
    
    return result_df


def predict_price_rise_or_drop(df, window_size=7, next_n_days=1):
    """
    根据历史涨跌幅分布，预测下一个交易日股价涨跌的概率
    
    1. 首先，计算涨跌幅的分布，分桶后得到每个桶的涨跌幅范围和占比
    2. 然后，只取分布的前后10%的涨跌幅范围，计算涨跌概率
    3. 最后，根据涨跌概率，只是这前后10%的值来预测下一个交易日股价涨跌的概率
    
    参数:
    df (pd.DataFrame): 包含股票数据的DataFrame
    window_size (int): 时间窗口大小，默认为7个交易日
    
    返回:
    pd.DataFrame: 包含预测概率的DataFrame
    float: 预测准确度
    """
    # 获取历史涨跌幅数据
    price_changes_df = analyze_price_changes(df, window_size)
    
    # 创建预测结果DataFrame
    predictions = []
    
    # 记录预测正确的次数
    correct_predictions = 0
    total_predictions = 0
    
    # 从第二个数据点开始预测（需要至少一个历史数据点）
    for i in range(1, len(price_changes_df)):
        current_row = price_changes_df.iloc[i]
        current_date = current_row['Current_Date']
        current_index = current_row['Current_Index']
        current_percent_change = current_row['Percent_Change']
        
        # 获取历史涨跌幅数据（截至当前日期之前的所有数据）
        historical_changes = price_changes_df.iloc[:i]
        
        # 计算历史涨跌幅分布
        hist_values = historical_changes['Percent_Change'].values
        
        # 对历史涨跌幅进行排序
        sorted_hist = sorted(hist_values)
        
        # 计算前10%和后10%的阈值
        if len(sorted_hist) >= 10:  # 确保有足够的数据点
            
            lower_threshold = hist_values.min() + (hist_values.max() - hist_values.min()) * 0.1
            upper_threshold = hist_values.min() + (hist_values.max() - hist_values.min()) * 0.9
            
            # 计算落在前10%和后10%的数据点数量
            lower_10_percent = sum(1 for x in hist_values if x <= lower_threshold)
            upper_10_percent = sum(1 for x in hist_values if x >= upper_threshold)
            
            # 计算占比
            lower_percentage = (lower_10_percent / len(hist_values)) * 100
            upper_percentage = (upper_10_percent / len(hist_values)) * 100
            
            print(f"历史涨跌幅分布分析，总的数据行数: {len(hist_values)}")
            print(f"  前10%涨跌幅范围: <= {lower_threshold:.2f}%, 占比: {lower_percentage:.2f}%")
            print(f"  后10%涨跌幅范围: >= {upper_threshold:.2f}%, 占比: {upper_percentage:.2f}%")
        else:
            print("历史数据不足，无法计算可靠的分布")
        
        # 计算历史涨跌幅的标准差，用于设定相似度阈值
        std_dev = historical_changes['Percent_Change'].std()
        print("std_dev = ", std_dev)
        # 设定相似度阈值为标准差的0.5倍
        similarity_threshold = 0.5 * std_dev if std_dev > 0 else 2.0
        
        # 确定是否当前涨跌幅在前10%或后10%的范围内
        is_in_extreme_range = False
        if len(sorted_hist) >= 10:
            lower_threshold = hist_values.min() + (hist_values.max() - hist_values.min()) * 0.1
            upper_threshold = hist_values.min() + (hist_values.max() - hist_values.min()) * 0.9
            
            # 检查当前涨跌幅是否在极端范围内（前10%或后10%）
            if current_percent_change <= lower_threshold or current_percent_change >= upper_threshold:
                is_in_extreme_range = True
        
        # 只有在极端范围内才进行预测
        if is_in_extreme_range:
            # 找出历史上与当前涨跌幅相似的情况
            similar_situations = historical_changes[
                (historical_changes['Percent_Change'] >= current_percent_change - similarity_threshold) &
                (historical_changes['Percent_Change'] <= current_percent_change + similarity_threshold)
            ]
            
            # 如果没有足够的相似情况，扩大阈值到标准差的1倍
            if len(similar_situations) < 3 and len(historical_changes) >= 3:
                similarity_threshold = 1.0 * std_dev if std_dev > 0 else 5.0
                similar_situations = historical_changes[
                    (historical_changes['Percent_Change'] >= current_percent_change - similarity_threshold) &
                    (historical_changes['Percent_Change'] <= current_percent_change + similarity_threshold)
                ]
            
            # 如果仍然没有足够的相似情况，使用所有历史数据
            if len(similar_situations) < 3:
                similar_situations = historical_changes
            
            # 查看这些相似情况后的下一个交易日是涨还是跌
            next_day_rises = 0
            next_day_drops = 0
            
            for idx in similar_situations.index:
                similar_row = similar_situations.loc[idx]
                similar_index = similar_row['Current_Index']
                
                # 确保不是最后一个数据点
                if similar_index + next_n_days < len(df):
                    next_day_price = df.iloc[similar_index + next_n_days]['Close']
                    current_day_price = df.iloc[similar_index]['Close']
                    
                    if next_day_price > current_day_price:
                        next_day_rises += 1
                    elif next_day_price < current_day_price:
                        next_day_drops += 1
            
            # 计算涨跌概率
            total_similar = next_day_rises + next_day_drops
            rise_probability = next_day_rises / total_similar * 100 if total_similar > 0 else 50
            drop_probability = next_day_drops / total_similar * 100 if total_similar > 0 else 50
            
            # 预测结果（概率大的为预测结果）
            predicted_direction = 'Rise' if rise_probability > drop_probability else 'Drop'
            
            # 获取实际结果（如果有下一个交易日的数据）
            actual_direction = 'Unknown'
            is_correct = None
            
            # 排除掉前window_size行数据，只有在极端范围内才计算准确率
            if current_index + 1 < len(df) and current_index >= window_size:
                next_day_price = df.iloc[current_index + 1]['Close']
                current_price = df.iloc[current_index]['Close']
                
                actual_direction = 'Rise' if next_day_price > current_price else 'Drop'
                is_correct = predicted_direction == actual_direction
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
        else:
            # 不在极端范围内，不进行预测
            predicted_direction = 'No Prediction'
            actual_direction = 'Unknown'
            is_correct = None
            rise_probability = 0
            drop_probability = 0
            similar_situations = pd.DataFrame()  # 空DataFrame
        
        predictions.append({
            'Date': current_date,
            'Index': current_index,
            'Current_Percent_Change': current_percent_change,
            'Similarity_Threshold': similarity_threshold,
            'Similar_Situations_Count': len(similar_situations),
            'Rise_Probability': round(rise_probability, 2),
            'Drop_Probability': round(drop_probability, 2),
            'Predicted_Direction': predicted_direction,
            'Actual_Direction': actual_direction,
            'Is_Correct': is_correct,
            'Is_In_Extreme_Range': is_in_extreme_range
        })
    
    # 创建预测结果DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # 计算预测准确度
    accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    
    print(f"预测完成! 时间窗口: {window_size}个交易日")
    print(f"预测准确度: {accuracy:.2f}%")
    print(f"总预测次数: {total_predictions}, 正确次数: {correct_predictions}")
    
    return predictions_df, accuracy



# 执行分析
if __name__ == "__main__":
    # 每年的预测精度都会有差别，需要根据实际情况调整
    df = load_data_with_date_range(file_path='TLT_2005-01-01_2025-02-21_1d.csv', 
                                   start_date='2024-01-01', 
                                   end_date='2025-03-03')
    # analyze_price_changes(df, window_size=3)
    # analyze_price_changes(df, window_size=5)
    # analyze_price_changes(df, window_size=7)
    # analyze_price_changes(df, window_size=14)
    df, acc = predict_price_rise_or_drop(df, window_size=3, next_n_days=2)
    df.to_csv('predictions.csv', index=False)
    





