import pymysql
import pymongo
import collections
import pandas as pd
from pymongo import MongoClient

#This list is only used for first version test
#test_devid = {'07420000000000A7','074200000000008B','0742000000000093','074200000000009D','0742000000000091'
#            '07420000000000A6','074200000000008A','0742000000000081','0742000000000096','07420000000000A2'}

test_devid = ['07420000000000A7']

host = 'yun.fenjin.cn'
host2 = '127.0.0.1'

#connect to MongoDB
client = MongoClient(host, 29916)
client2 = MongoClient(host2, 27017)
db = client.cloud
db2 = client2.cloud  # mydb数据库，同上解释
#use userid and password to login
db.authenticate("cloud", "cloud2018")
#choose historydatas as collection
historydataCollection = db.historydatas
equipmentRunTimeDaysCollection = db2.test1
room_avg_temp = {}

def connect_mysql():
    conn = pymysql.connect(host='180.76.247.84', user='root', passwd='pass9cuo@2018', db='cloud',charset='utf8')
    cursor = conn.cursor()
    sql = 'SELECT SensorNo FROM Sensors'
    try:
        #execute sql query
        cursor.execute(sql)
        #record the result
        row = cursor.fetchall()
        for devid in test_devid:
            process_data(devid)
    except:
        print("Error:unable to fetch data")
    conn.close()

def process_data(devid):        
    df1 = find_in_mongodb(devid)
    sensor_avg_dict = create_senseor_avg_dict(df1)
    sensor_avg_pd = pd.DataFrame(data=sensor_avg_dict,index=df1['date'].tolist())
    sensor_avg_pd.to_csv("./data.csv",index=True)

def find_in_mongodb(devid):
    #find the data according to given devid
    result = historydataCollection.find({"devid": devid})
    #change the table to pandas dataframe
    df = pd.DataFrame(list(result))
    #ignore other information but focus on history and form a new panads dataframe which contains date and hisdata
    df1 = df.loc[:,['date','hisdata']].copy()
    #choose the first value in first data,return the tuple of values
    return df1

def create_senseor_avg_dict(df):
    sensor_avg_dict = {}
    for index in range(9):
        temp_list = []
        for date in range(len(df.loc[:,'hisdata'])):
            temp_avg = create_sensor_dict(df.loc[date,'hisdata'],index)
            temp_list.append(temp_avg)
        sensor_avg_dict[index] = temp_list
    return sensor_avg_dict

def create_sensor_dict(hisdata,sensornum):
    avg = 0
    temp = 0
    for index,item in enumerate(hisdata):
        for vale in item['d']:
            if vale['c'] == 33 or (vale['i'] != sensornum and vale['c'] != 33):
                continue
            temp = float(vale['v'])   
        avg += temp
        if index == (len(hisdata) - 1):
            avg /= (index+1)
    return avg

connect_mysql()
