import time
import json
import re
import threading
from datetime import datetime

import asyncio
import websockets
import pandas as pd

import settings
LOC_DUMP=setting.okcoinAPI['loc_dump_data']
URL=setting.okcoinAPI['url']
URL_SPOT=setting.okcoinAPI['url_spot']

kline_columns=['time', 'open', 'highest', 'lowest', 'close', 'volume', 'amount']
depth_asks_columns=['asks_Price', 'asks_Amount(Contract)', 'asks_Amount(Coin)','asks_Cumulant(Coin)','asks_Cumulant(Contract)']
depth_bids_columns=['bids_Price', 'bids_Amount(Contract)', 'bids_Amount(Coin)','bids_Cumulant(Coin)','bids_Cumulant(Contract)']
trade_columns=['tid', 'price', 'amount', 'time', 'type', 'amount(coin)']

#return : dict, int
async def ticker_handler(dd, i):
    df=pd.DataFrame(dd['data'],index=[i])
    return df, i+len(df)
    
#return : list, int, list
async def kline_handler(dd, i):
    d_kline =pd.DataFrame(dd['data'], index=range(len(dd['data'])), columns=kline_columns)
    
    d_kline.index += i
    return d_kline, i+len(d_kline)

#return : list, int, list.list
async def depth_handler(dd, i):
    d_asks =pd.DataFrame(dd['data']['asks'], index=range(len(dd['data']['asks'])), columns=depth_asks_columns)
    d_bids =pd.DataFrame(dd['data']['bids'], index=range(len(dd['data']['bids'])), columns=depth_bids_columns)
    df=pd.concat([d_asks, d_bids], axis=1)
    df["time"]=dd['data']["timestamp"]
    
    df.index += i
    return df, i+len(df)
    
#return : list, int, list
async def trade_handler(dd, i):
    d_trade =pd.DataFrame(dd['data'], index=range(len(dd['data'])), columns=trade_columns)
    
    d_trade.index += i
    return d_trade, i+len(d_trade)
    
#return : str, int, str, list
async def data_handler(data, dict_i, dd_type=None):
    try:
        dd=json.loads(data)[0]
        if 'errorcode' in dd or 'result' in dd['data']:
            print(dd)
            return pd.DataFrame(),None,None
            
        channel = dd['channel']
        i=dict_i[channel]
        
        if not dd_type:
            for j in ['ticker','kline','depth','trade']:
                if re.search(j, channel) != None:
                    dd_type = j
                    break

        if dd_type == 'ticker':
            df, i = await ticker_handler(dd, i)
            return df, i, channel
        elif dd_type == 'kline':
            df, i = await kline_handler(dd, i)
            return df, i, channel
        elif dd_type == 'depth':
            df, i = await depth_handler(dd, i)
            return df, i, channel
        elif dd_type == 'trade':
            df, i = await trade_handler(dd, i)
            return df, i, channel
    except:
        print(data)
        return pd.DataFrame(),None,None

#return : DataFrame, str, str
async def dump_handler(df0, channel, type_dump='csv', locate=r'H:/'):
    if type_dump == 'test':
        with open(locate+channel, 'a') as fp:
            fp.write(df0)
    if type_dump == 'csv':
        df0.to_csv(r'%s%s%d.csv'%(locate, channel, time.time()))
    if type_dump == 'mysql':
        pass
    
    
async def time_handler(df0_time, flag_utc=None, type_time='int'):
    if not flag_utc:
        if type_time=='str':
            df0_time = df0_time.apply(lambda x: eval(x[:10]))
        if type_time=='int':
            df0_time = df0_time.apply(lambda x: int(str(x)[10:]))
    if flag_utc == 'utc':
        df0_time = df0_time.apply(lambda x: datetime.strftime(datetime.utcfromtimestamp(x),"%Y-%m-%d %H:%M:%S"))
    return df0_time
    
#flag_group[0] equal main group, flag_group[1] equal second group, same as continue
async def multi_handler(df0, flag_group=['time'], flag_filter='last', mul_obj=None, window=None):
    if flag_filter == 'first':
        df1=df0.groupby(flag_group).pipe(lambda x: x.head(1))
            
    if flag_filter == 'last':
        df1=df0.groupby(flag_group).pipe(lambda x: x.tail(1))
            
    if flag_filter == 'sma':
        if not mul_obj or not window:
            df1=df0.groupby(flag_group).pipe(lambda x: x.rolling(min_periods=1,window=window).mean())
        if mul_obj and window:
            df1=df0.groupby(flag_group).pipe(lambda x: x[mul_obj].rolling(min_periods=1,window=window).mean())
            
    if flag_filter == 'ema':
        if not mul_obj or not window:
            df1=df0.groupby(flag_group).pipe(lambda x: x.ewm(min_periods=1,alpha=1./window).mean())
        if mul_obj and window:
            df1=df0.groupby(flag_group).pipe(lambda x: x[mul_obj].ewm(min_periods=1,alpha=1./window).mean())
            
    df1.index=range(len(df1))
    return df1
    
#websockets send/recv
async def get_data(url, list_channel, type_dump='csv', locate_dump=r'H:/', NUM_TO_DUMP=0, TIME_TO_DUMP=0, TIME_TO_PING=30):
    dict_df0={}
    dict_i={}
    t0=time.time()
    td=time.time()
    
    try:
        async with websockets.connect(url) as ws:
            
            for channel in list_channel:
                msg = "{'event':'addChannel','channel':'%s'}"%channel
                dict_df0[channel] = pd.DataFrame()
                dict_i[channel] = 0
                await ws.send(msg)

            data = await ws.recv()
            while data:
                df, i, d_channel = await data_handler(data, dict_i, None)
                if d_channel:
                    dict_df0[d_channel]=dict_df0[d_channel].append(df)
                    dict_i[d_channel]=i
                data = await ws.recv()
                

                if TIME_TO_DUMP and int(time.time()-td)>=TIME_TO_DUMP:
                    for j in dict_df0:
                        await dump_handler(dict_df0[j], j, type_dump, locate_dump)
                        dict_df0[j]=pd.DataFrame()
                        dict_i[j]=0
                    td=time.time()
                if NUM_TO_DUMP:
                    for k in dict_i:
                        if NUM_TO_DUMP<=dict_i[k]:
                            await dump_handler(dict_df0[k], k, type_dump, locate_dump)
                            dict_df0[k]=pd.DataFrame()
                            dict_i[k]=0
                if TIME_TO_PING <= int(time.time()-t0):
                    await ws.send("{'event':'Ping'}")
                    t0=time.time()

    except Exception as e:
        print(e)
    finally:
        pass
        #await dump_handler(df0, type_dump)

        

def task_gen(channel,X='btc',Y='this_week',Z='1min',type2handler=None,type2dump='csv'):
    list_channel=[]
    
    if type2handler == None:
        for i in ['ticker','kline','depth','trade']:
            if re.search(i, channel) != None:
                type2handler = i
                break

    f_split = lambda _: re.split(r'[, ]+', _)
    if type(X)==str:
        X,Y,Z = map(f_split, [X,Y,Z])
        
    for i in X:
        for j in Y:
            for k in Z:
                channel_s = re.sub('X',i,channel)
                channel_s = re.sub('Y',j,channel_s)
                channel_s = re.sub('Z',k,channel_s)
                list_channel.append(channel_s)
    
    return list_channel, type2handler
        
        
def test():
    url = URL   #OKEx Contract url 
    url_spot = URL_SPOT
    
    t_channel, t_type= task_gen(
    channel = 'k_sub_futureusd_X_ticker_Y',
    X='btc, ltc, eth, etc, bch', 
    Y='this_week next_week quarter')
    
    k_channel, k_type = task_gen(
    channel = 'ok_sub_futureusd_X_kline_Y_Z',
    X='btc, eth, etc', 
    Y='this_week next_week quarter', 
    Z='1min 3min 5min')
    

    d_channel, d_type= task_gen(
    channel = 'ok_sub_futureusd_X_depth_Y_Z',
    X='btc', 
    Y='this_week next_week', 
    Z='20')
    

    tr_channel, tr_type= task_gen(
    channel = 'ok_sub_futureusd_X_trade_Y',
    X='btc ltc', 
    Y='this_week next_week')
    
    list_channel=k_channel#+d_channel#+tr_channel
    
    try:
        t=time.time()
        task=[get_data(url, list_channel, TIME_TO_DUMP=1800, type_dump='csv', locate_dump=LOC_DUMP)]
        asyncio.get_event_loop().run_until_complete(asyncio.wait(task))
    finally:
        print(time.time()-t)
