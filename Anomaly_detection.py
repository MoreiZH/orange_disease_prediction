#!usr/bin/python3
# -*- coding: utf-8 -*- 
"""
Project: disease_weather_analysis
File: anomaly_detection_zscore.py
IDE: PyCharm
Creator: morei
Email: zhangmengleiba361@163.com
Create time: 2021-03-17 15:57
Introduction:
"""


import pandas as pd
import time
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


def timestamp_to_date(timestamp):
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    time_l = time.localtime(timestamp)
    y_m = time.strftime("%Y-%m", time_l)
    return y_m


def zscore_check(dataframe, colname, threshold=3):
    se = dataframe[colname]
    zscore = (se - se.mean()) / (se.std())
    return dataframe[zscore.abs() < threshold]


def zscore_mad_check(dataframe, colname, threshold=3.5):
    se = dataframe[colname]
    MAD = (se - se.median()).abs().median()
    zscore = ((se - se.median()) * 0.6475 / MAD).abs()
    return dataframe[zscore < threshold]


if __name__ == "__main__":
    # input data
    data = pd.read_csv('./output2/disease_weather_province100_d_tem.csv')
    prov_list = list(set(data['province'].tolist()))
    # 异常值检测
    df1 = zscore_check(data, 'avg_tem')
    # df1.to_csv('./output2/data_weather_province100_filtered_avg_tem_zscore.csv', index=False, encoding='utf-8-sig')
    df2 = zscore_mad_check(data, 'avg_tem')
    df2.to_csv('./output2/data_weather_province100_filtered_avg_tem_mad_zscore.csv', index=False, encoding='utf-8-sig')
    df3 = zscore_mad_check(df2, 'avg_rhu').drop_duplicates()
    df3.to_csv('./output2/data_weather_province100_filtered_avg_tem_avg_rhu_mad_zscore.csv', index=False, encoding='utf-8-sig')
    # 按省提取数据
    for yr in [2019, 2020, 2021]:
        temp1 = df2[df2['年'] == yr]
        out1 = './output2/weather_avg_tem_mad_zscore_{}.csv'.format(yr)
        temp1.to_csv(out1, index=False, encoding='utf-8-sig')
        for prov in prov_list:
            temp2 = temp1[temp1['province'] == prov]
            out2 = './output2/weather_avg_tem_mad_zscore_{}{}.csv'.format(prov, yr)
            temp2.to_csv(out2, index=False, encoding='utf-8-sig')
    # all
    # input data
    data = pd.read_csv('./output2/disease_weather_province_all.csv')
    # 异常值检测
    df2 = zscore_mad_check(data, 'avg_tem')
    df2.to_csv('./output2/data_weather_province_all_filtered_avg_tem_mad_zscore.csv', index=False, encoding='utf-8-sig')

    a = 1