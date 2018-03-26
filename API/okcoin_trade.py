# coding: utf-8
import hashlib
import asyncio

import pandas as pd
import numpy as np

from ..settings import okcoinAPI
api_Key = okcoinAPI['api_Key']
secret_Key = okcoinAPI['secret_Key']


#business
def build_Sign(params,secret_Key):
    sign = ''
    for key in sorted(params.keys()):
        sign += key + '=' + str(params[key]) +'&'
    return  hashlib.md5((sign+'secret_key='+secret_Key).encode("utf-8")).hexdigest().upper()

def str_login(api_key, secret_Key):
    params={
    'api_key':api_key
    }
    
    if 'sign' not in params:
        params['sign'] = build_Sign(params, secret_Key)
    return '{"event":"login","parameters":%s}'%(str(params))

def str_userinfo(api_key, secret_Key, type_trade="futures"):
    if type_trade == "spot":
        channel = 'ok_spot_userinfo'
    if type_trade == "futures":
        channel = 'ok_futureusd_userinfo'
    params={
    'api_key':api_key
    }
    
    if 'sign' not in params:
        params['sign'] = build_Sign(params, secret_Key)
    return "{'event':'addChannel','channel':'%s','parameters':%s}"%(channel, str(params))

def str_futures_orderinfo(api_key, secret_Key, futures_symbol, futures_Type, order_id, status='1', current_page='1',page_length='1'):
    params={
    'api_key':api_key,
    'sign': '',
    'symbol': futures_symbol,
    'order_id': order_id,
    'contract_type': futures_Type,
    'status': status,
    'current_page': current_page,
    'page_length': page_length
    }
    
    if not params['sign']:
        params['sign'] = build_Sign(params, secret_Key)
    return "{'event':'addChannel','channel':'ok_futureusd_orderinfo','parameters':%s}"%(str(params))
    
def str_spot_orderinfo(api_key, secret_Key, order_symbol='', order_id=''):
    params={
    'api_key':api_key,
    'sign': '',
    'symbol': order_symbol,
    'order_id': order_id
    }
    
    if not params['sign']:
        params['sign'] = build_Sign(params, secret_Key)
    return "{'event':'addChannel','channel':'ok_spot_orderinfo','parameters':%s}"%(str(params))

#Futures trade
def str_Futures(api_key,secret_Key,futures_symbol,futures_Type,trade_Type,price='',amount='',match_price='0',lever_rate='10'):
    params={
        'api_key': api_Key,
        'sign' : '',
        'symbol': futures_symbol,
        'contract_type': futures_Type,
        'price': price,
        'amount': amount,
        'type': trade_Type,
        'match_price': match_price,
        'lever_rate': lever_rate
    }
    if not params['sign']:
        params['sign'] = build_Sign(params, secret_Key)
    
    finalStr =  "{'event':'addChannel','channel':'ok_futureusd_trade','parameters':%s,'binary':'true'}"%(str(params))
    return finalStr
    
def str_Futures_cancel(api_key,secret_Key,channel,futures_symbol,futures_Type,order_id):
    params={
        'api_key': api_Key,
        'sign' : '',
        'symbol': futures_symbol,
        'order_id': order_id,
        'contract_type': futures_Type
    }
    if not params['sign']:
        params['sign'] = build_Sign(params, secret_Key)
    
    finalStr =  "{'event':'addChannel','channel':'ok_futureusd_cancel_order','parameters':%s,'binary':'true'}"%(str(params))
    return finalStr
    
def str_Order(api_key,secret_Key,order_symbol,order_Type,price='',amount=''):
    params={
        'api_key': api_Key,
        'sign' : '',
        'symbol': order_symbol,
        'type': order_Type,
        'price': price,
        'amount':amount
    }
    if not params['sign']:
        params['sign'] = build_Sign(params, secret_Key)
    
    finalStr =  "{'event':'addChannel','channel':'ok_spot_order','parameters':%s,'binary':'true'}"%(str(params))
    return finalStr
    
def str_Order_cancel(api_key,secret_Key,order_symbol,order_id):
    params={
        'api_key': api_Key,
        'sign' : '',
        'symbol': order_symbol,
        'order_id': order_id
    }
    if not params['sign']:
        params['sign'] = build_Sign(params, secret_Key)
    
    finalStr =  "{'event':'addChannel','channel':'ok_spot_cancel_order','parameters':%s,'binary':'true'}"%(str(params))
    return finalStr
