import warnings
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
data = pd.read_csv('data.csv',index_col=0,date_parser=dateparse)
#选择不同的传感器来进行预测
series = data['0']
split_point = len(series) - 7
dataset,validation = series[0:split_point],series[split_point:]
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

#原始数据摘要
#series = Series.from_csv('dataset.csv')
#print(series.describe())

#原始数据可视化

#1.折线图
#series.plot()
#pyplot.show()

#2.密度图
#pyplot.figure(1)
#pyplot.subplot(211)
#series.hist()
#pyplot.subplot(212)
#series.plot(kind='kde')
#pyplot.show()

#差分化函数
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return pd.Series(diff)

#测试稳定性
def test_stationary(X):
    series = pd.Series.from_csv('dataset.csv')
    X = series.values
    X = X.astype('float32')
    stationary = difference(X)
    result = adfuller(stationary)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    # 绘制平稳时序的图
    stationary.plot()
    pyplot.show()
    # 保存
    stationary.to_csv('stationary.csv')

# 基于给定的ARIMA(p,d,q)值评评估该模型并返回RMSE值
def evaluate_arima_model(X, arima_order):
    # 准备训练集数据
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # 作出预测
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        # model_fit = model.fit(disp=0)
        try:
            model_fit = model.fit(disp=-1)
        except:
            print("wrong here!!!")
        try:
            yhat = model_fit.forecast()[0]
        except:
            print("wrong here too!!!")
        predictions.append(yhat)
        history.append(test[t])
    # 计算测试集上的预测误差
    try:
        mse = mean_squared_error(test, predictions)
    except:
        print(mse)
    rmse = math.sqrt(mse)
    print(rmse)
    return rmse
 
# 评估带有不同参数值的每个ARIMA模型
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA: %s RMSE: %.3f' % (order,mse))
                except:
                    print("here is some exception")
                    continue
    print('Best ARIMA %s RMSE: %.3f' % (best_cfg, best_score))
 
# 评价参数
def test_parameters():
    p_values = range(0, 5)
    d_values = range(0, 3)
    q_values = range(0, 5)
    warnings.filterwarnings("ignore")
    series = pd.Series.from_csv('dataset.csv')
    evaluate_models(series.values, p_values, d_values, q_values)
    #通过网格查询，我们可以得知ARIMA模型参数选择是(4,0,0)
#test_parameters()

#计算残差
def residual_calculation():
    series = pd.Series.from_csv('dataset.csv')
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # 前向验证
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # 预测
        model = ARIMA(history, order=(4,0,0))
        model_fit = model.fit(disp=-1)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # 观测
        obs = test[i]
        history.append(obs)
    # 误差
    residuals = [test[i]-predictions[i] for i in range(len(test))]
    residuals = pd.DataFrame(residuals)
    print(residuals.describe())
    pyplot.figure()
    pyplot.subplot(211)
    residuals.hist(ax=pyplot.gca())
    pyplot.subplot(212)
    residuals.plot(kind='kde', ax=pyplot.gca())
    pyplot.show()

#residual_calculation()
#通过计算残差，我们可以知道残差仍有一个均值为0.247989。所以残差不为0，我们仍需要调整模型

def modified_model():
    # prepare data
    series = pd.Series.from_csv('dataset.csv')
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    bias = 0.247989
    for i in range(len(test)):
        # predict
        model = ARIMA(history, order=(4,0,0))
        model_fit = model.fit(disp=-1)
        yhat = bias + float(model_fit.forecast()[0])
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # report performance
    mse = mean_squared_error(test, predictions)
    rmse = math.sqrt(mse)
    print('RMSE: %.3f' % rmse)
    # 总结残差
    residuals = [test[i]-predictions[i] for i in range(len(test))]
    residuals = pd.DataFrame(residuals)
    print(residuals.describe())
    # 绘制残差
    pyplot.figure()
    pyplot.subplot(211)
    residuals.hist(ax=pyplot.gca())
    pyplot.subplot(212)
    residuals.plot(kind='kde', ax=pyplot.gca())
    pyplot.show()
#modified_model()

#保存模型
# RRIMA型模型的monkey补丁
def __getnewargs__(self):
    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
 
ARIMA.__getnewargs__ = __getnewargs__
 
def save_model():
    # 准备数据
    series = pd.Series.from_csv('dataset.csv')
    X = series.values
    X = X.astype('float32')
    # 拟合模型
    model = ARIMA(X, order=(4,0,0))
    model_fit = model.fit(disp=-1)
    # bias constant, could be calculated from in-sample mean residual
    bias = 0.247989
    # 保存模型
    model_fit.save('model.pkl')
    np.save('model_bias.npy', [bias])
#save_model()

#预测
def prediction():
    model_fit = ARIMAResults.load('model.pkl')
    bias = np.load('model_bias.npy')
    yhat = bias + float(model_fit.forecast()[0])
    print('Predicted: %.3f' % yhat)
#prediction()

#进一步滚动预测
def more_prediction():
    dataset = pd.Series.from_csv('dataset.csv')
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = pd.Series.from_csv('validation.csv')
    y = validation.values.astype('float32')
    # load model
    model_fit = ARIMAResults.load('model.pkl')
    bias = np.load('model_bias.npy')
    # 做出第一个预测
    predictions = list()
    yhat = bias + float(model_fit.forecast()[0])
    predictions.append(yhat)
    history.append(y[0])
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
    # 滚动预测
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(4,0,0))
        model_fit = model.fit(disp=-1)
        yhat = bias + float(model_fit.forecast()[0])
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    # 报告性能
    mse = mean_squared_error(y, predictions)
    rmse = math.sqrt(mse)
    print('RMSE: %.3f' % rmse)
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()
more_prediction()

def test():
    # load model
    model_fit = ARIMAResults.load('model.pkl')
    forecast = model_fit.forecast(steps=7)[0]
    print(forecast)
#test()
