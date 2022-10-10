# to be schedule in a cron job with GitHub actions

import requests                    # for "get" request to API
import json                        # parse json into a list
import pandas as pd                # working with data frames
import datetime as dt              # working with dates
#import os
#import time
#from threading import Thread

# this function allow to get the data from binance on EST time

def get_binance_bars(symbol, interval, startTime, endTime):
 
    url = "https://api.binance.com/api/v3/klines"
 
    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'
 
    req_params = {"symbol" : symbol, 'interval' : interval, 'startTime' : startTime, 'endTime' : endTime, 'limit' : limit}
 
    df = pd.DataFrame(json.loads(requests.get(url, params = req_params).text))
 
    if (len(df.index) == 0):
        return None
     
    df = df.iloc[:, 0:6]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
 
    df.open      = df.open.astype("float")
    df.high      = df.high.astype("float")
    df.low       = df.low.astype("float")
    df.close     = df.close.astype("float")
    df.volume    = df.volume.astype("float")
     
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.datetime]

    df.index = df.index - pd.to_timedelta(4, unit="h")
 
    return df

import os.path

if not os.path.isfile('BTCUSDT_historical.csv'):
    print ("File not exist")
    start_time = dt.datetime(2017, 6, 17)
    df = pd.DataFrame()

else:
    print ("File exist")
    # read the data previously store in your directory
    df = pd.read_csv("BTCUSDT_historical.csv", index_col="Unnamed: 0", parse_dates=True)
    # get the last index value
    start_time = df.index[-5]

now_time = dt.datetime.now()

while True:
  next = start_time + pd.to_timedelta(1000, unit="h")
  btc = get_binance_bars("BTCUSDT", "1h", start_time, now_time)
  print(btc.index[-1])
  df = pd.concat([df, btc])
  
  if btc.index[-1].date() == now_time.date():
    break
  start_time = btc.index[-1]

df.sort_index(inplace=True)
df = df[~df.index.duplicated(keep='last')]
df.to_csv("BTCUSDT_historical.csv")
