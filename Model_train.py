#!usr/bin/python3
# -*- coding: utf-8 -*- 
"""
Project: disease_weather_analysis
File: ols.py
IDE: PyCharm
Creator: morei
Email: zhangmengleiba361@163.com
Create time: 2021-03-16 16:32
Introduction:
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import csv
import warnings
from sklearn.metrics import mean_squared_error as mse

warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif'] = ['SimHei']


def fun_l(x, A, B):
    return A * x + B


def fun_exp_p(x, a, b, c):
    return a * np.exp(-b * x) + c


def fun_exp(x, a, b):
    return x ** a + b


# input data
train = pd.read_excel('./data/weather_avg_tem_mad_zscore_广西201903-202002.xls')
test = pd.read_excel('./data/weather_avg_tem_mad_zscore_广西202003-202102.xls')
province = train['province'].tolist()[0]
train = train.sort_values(by='ym')
test = test.sort_values(by='ym')
feats = ['avg_tem', 'precipitation']
# train = train[feats]
train['频率'] = train.频次.apply(lambda x: x/train['频次'].sum())
test['频率'] = test.频次.apply(lambda x: x/test['频次'].sum())

# zscore standarlization
train[feats] =  train[feats].apply(lambda x: preprocessing.scale(x))
test[feats] =  test[feats].apply(lambda x: preprocessing.scale(x))
train = train[['频率'] + feats]
test = test[['频率'] + feats]
# train = train.round(0)
trainy = train['频率']

# check correlation
# datacor = np.corrcoef(train,rowvar=0)
# datacor = pd.DataFrame(data=datacor,columns=train.columns,index=train.columns)
# plt.figure(figsize=(8,8))
# ax = sns.heatmap(datacor,square=True,annot=True,fmt=".3f",linewidths=.5,cmap="YlGnBu",cbar_kws={"fraction":0.046,"pad":0.03})
# ax.set_title("数据变量相关性")
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.show()

# 单因素分析
# train
train1 = train[['频率', 'avg_tem']]
train_df = train1.groupby(by=['avg_tem'])['频率'].sum().to_frame()
train_df['avg_tem'] = train_df.index
train_df = train_df.reset_index(drop=True)
y = train_df['频率']
x = train_df['avg_tem']
# test
test1 = test[['频率', 'avg_tem']]
test_df = test1.groupby(by=['avg_tem'])['频率'].sum().to_frame()
test_df['avg_tem'] = test_df.index
test_df = test_df.reset_index(drop=True)
ty = test_df['频率']
tx = test_df['avg_tem']
model_2019_list = [('province', 'model_fit', 'R2_train', 'mse_train', 'R2_2020', 'mse_2020')]
# # plt.scatter(x, y)
# # plt.show()

# 线性拟合
# 2019
popt1, pcov1 = curve_fit(fun_l, x, y)
# popt数组中，三个值分别是待求参数a,b,c
a1 = popt1[0].round(4)
b1 = popt1[1].round(4)
model_l_2019 = 'y = {}*x + ({})'.format(a1, b1)
y1 = [fun_l(i, popt1[0], popt1[1]) for i in x]
r2_2019 = r2_score(y, y1).round(4)
mse2019 = mse(y, y1).round(4)
# predict 2020
y_pre = [fun_l(i, popt1[0], popt1[1]) for i in tx]
r2_2019_pre = r2_score(ty, y_pre).round(4)
mse2019_pre = mse(ty, y_pre).round(4)
model_2019_list.append((province, model_l_2019, r2_2019, mse2019, r2_2019_pre, mse2019_pre))

# 多项式拟合
z1 = np.polyfit(x, y, 2)  # 曲线拟合，返回值为多项式的各项系数
p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
print(p1)
model_m2_2019 = 'y = {}*x2 + {}*x + ({})'.format(p1[0].round(4), p1[1].round(4), p1[2].round(4))
y_m2 = p1(x)
r2_m2_2019 = r2_score(y, y_m2).round(4)
mse_m2_2019 = mse(y, y_m2).round(4)
# predict 2020
y_m2_pre = p1(tx)
r2_m2_2019_pre = r2_score(ty, y_m2_pre).round(4)
mse_m2_2019_pre = mse(ty, y_m2_pre).round(4)
model_2019_list.append((province, model_m2_2019, r2_m2_2019, mse_m2_2019, r2_m2_2019_pre, mse_m2_2019_pre))

# 指数拟合
# 2019
popt_e, pcov_e = curve_fit(fun_exp, x, y)
# popt数组中，三个值分别是待求参数a,b,c
a_e1 = popt_e[0].round(4)
b_e1 = popt_e[1].round(4)
model_e_2019 = 'y = x**{} + ({})'.format(a_e1, b_e1)
y_e = [fun_exp(i, popt_e[0], popt_e[1]) for i in x]
r2_e_2019 = r2_score(y, y_e).round(4)
mse_e_2019 = mse(y, y_e).round(4)
# predict 2020
y_e_pre = [fun_exp(i, popt_e[0], popt_e[1]) for i in tx]
r2_e_2019_pre = r2_score(ty, y_e_pre).round(4)
mse_e_2019_pre = mse(ty, y_e_pre).round(4)
model_2019_list.append((province, model_e_2019, r2_e_2019, mse_e_2019, r2_e_2019_pre, mse_e_2019_pre))

# multivariate
formula = "频率 ~ avg_tem+precipitation"
lm = smf.ols(formula, train).fit()
intercept = lm.params[0].round(4)
平均温度_p = lm.params[1].round(4)
precp_p = lm.params[2].round(4)
model_lm_2019 = 'y = avg_tem*{} + precipitation*{}+ ({})'.format(平均温度_p, precp_p, intercept)
ttt = lm.summary()
x2019 = train[['avg_tem', 'precipitation']]
x2019 = sm.add_constant(x2019)
lm_y_2019 = lm.predict(x2019)
r2_lm_2019 = r2_score(y, lm_y_2019).round(4)
mse_lm_2019 = mse(y, lm_y_2019).round(4)
x2020 = test[['avg_tem', 'precipitation']]
x2020 = sm.add_constant(x2020)
lm_y_2020 = lm.predict(x2020)
r2_lm_2020 = r2_score(ty, lm_y_2020).round(4)
mse_lm_2020 = mse(ty, lm_y_2020).round(4)
model_2019_list.append((province, model_lm_2019, r2_lm_2019, mse_lm_2019, r2_lm_2020, mse_lm_2020))

# model output to file
with open('./output2/models-frequency-ratio/models_frequency_ratio_{}_201903-202002_zscore_avg_tem.csv'.format(province), 'w',
          newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(model_2019_list)
a = 1
