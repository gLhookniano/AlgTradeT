# coding: utf-8
import numpy as np
import pandas as pd
import talib as ta


def ATR(df0, window=14):
    """ 
    ATR(df0, window=14)
    
    return: <float>list
    """
    return ta.ATR(np.array(df0.highest), np.array(df0.lowest), np.array(df0.close), timeperiod=window)
    
    
def BB(df0, window=14, flag_dev_u=2, flag_dev_d=2, flag_source='close', flag_ma='sma'):
    if flag_ma=='sma':
        matype=0
    if flag_ma=='ema':
        matype=1
    UB, MB, DB = ta.BBANDS(np.array(df0[flag_source]), timeperiod=window, matype=matype, nbdevup=flag_dev_u, nbdevdn=flag_dev_d)
    return UB, MB, DB
    
def BBX(df0, U_window=42, M_window=30, D_window=21, flag_dev_u=2, flag_dev_d=2, flag_source='close', flag_ma='sma'):
    """
    BB trendCapture
    """
    if flag_ma=='sma':
        matype=0
    if flag_ma=='ema':
        matype=1
    U_UB, U_MB, U_DB = ta.BBANDS(np.array(df0[flag_source]), timeperiod=U_window, matype=matype, nbdevup=flag_dev_u, nbdevdn=flag_dev_d)
    M_UB, M_MB, M_DB = ta.BBANDS(np.array(df0[flag_source]), timeperiod=M_window, matype=matype, nbdevup=flag_dev_u, nbdevdn=flag_dev_d)
    D_UB, D_MB, D_DB = ta.BBANDS(np.array(df0[flag_source]), timeperiod=D_window, matype=matype, nbdevup=flag_dev_u, nbdevdn=flag_dev_d)
    return U_UB, M_MB, D_DB
    
    
def DC(df0, window=20):
    U = df0.highest.rolling(window=window).max()
    D = df0.lowest.rolling(window=window).min()
    return np.array(U), np.array(D)
    
def DCX(df0, U_window=42, D_window=21):
    """
    DC trendCapture
    """
    U = df0.highest.rolling(window=U_window).max()
    D = df0.lowest.rolling(window=D_window).min()
    return np.array(U), np.array(D)

def EMA(df0, window=14, flag_dat_obj='close'):
    return ta.EMA(np.array(df0[flag_dat_obj]), timeperiod=window)
    
def MA(df0, window=14, flag_dat_obj='close', flag_ma='sma'):
    if flag_ma=='sma':
        matype=0
    if flag_ma=='ema':
        matype=1
    return ta.MA(np.array(df0[flag_dat_obj]), timeperiod=window, matype=matype)

def MACD(df0, win_fast=12, win_slow=26, win_sign=9, flag_dat_obj='close', flag_DI=False):
    if flag_DI: #Demand Index
        nd = np.array((df0.close*2+df0.highest+df0.lowest)/4.)
    if not flag_DI:
        nd = np.array(df0[flag_dat_obj])
    MACD, SIG, HIST = ta.MACD(nd, fastperiod=win_fast, slowperiod=win_slow, signalperiod=win_sign)
    return HIST, MACD, SIG

def RSI(df0, window=14, flag_dat_obj='close'):
    if flag_dat_obj:
        nd = np.array(df0[flag_dat_obj])
    return ta.RSI(nd, timeperiod=window)
        
def SAR(df0, AF_increse=0.02, AF_max=0.2):
    return ta.SAR(np.array(df0.highest), np.array(df0.lowest), acceleration=AF_increse, maximum=AF_max)
        
def SARX(ndh, ndl, init_val, tend, init_index=0, rev_val_per=0.002, AF_init=0.02, AF_increse=0.02, AF_max=0.2):
    if tend=='long' or tend:
        sarx = ta.SAREXT(ndh, ndl, startvalue=init_val, offsetonreverse=rev_val_per, accelerationinitlong=AF_init, accelerationlong=AF_increse, accelerationmaxlong=AF_max, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
        return sarx
    if tend=='short' or not tend:
        sarx = ta.SAREXT(ndh, ndl, startvalue=init_val, offsetonreverse=rev_val_per, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=AF_init, accelerationshort=AF_increse, accelerationmaxshort=AF_max)
        return sarx

def SRSI(df0, window=9, win_k=9, win_d=3, flag_ma='sma', flag_dat_obj='close'):
    if flag_ma=='sma':
        matype=0
    if flag_ma=='ema':
        matype=1
    if flag_dat_obj:
        nd = np.array(df0[flag_dat_obj])
    K, D = ta.STOCHRSI(nd, timeperiod=window, fastk_period=win_k, fastd_period=win_d, fastd_matype=matype)
    return K, D

def KD(df0, win_k_f=5, win_k=3, win_d=3, flag_ma='sma'):
    if flag_ma=='sma':
        matype=0
    if flag_ma=='ema':
        matype=1
    ndh = np.array(df0.highest)
    ndl = np.array(df0.lowest)
    ndc = np.array(df0.close)
    K, D = ta.STOCH(ndh, ndl, ndc, fastk_period=win_k_f, slowk_period=win_k, slowk_matype=matype, slowd_period=win_d, slowd_matype=matype)
    return K, D
    
def WR(df0, window=14):
    ndh=np.array(df0.highest)
    ndl=np.array(df0.lowest)
    ndc=np.array(df0.close)
    return ta.WILLR(ndh, ndl, ndc, timeperiod=window)
    
def TRIMA(df0, window=25, flag_dat_obj='close'):
    if flag_dat_obj:
        nd=np.array(df0[flag_dat_obj])
    return ta.TRIMA(nd, timeperiod=window)
    
def TRIX(df0, window=25, flag_dat_obj='close'):
    if flag_dat_obj:
        nd=np.array(df0[flag_dat_obj])
    return ta.TRIX(nd, timeperiod=window)
    
def MATRIX(df0, win_mt=25, win_ma=14, flag_dat_obj='close'):
    if flag_dat_obj:
        nd=np.array(df0[flag_dat_obj])
    trix = ta.TRIX(nd, timeperiod=win_mt)
    matrix = ta.MA(trix, timeperiod=win_ma)
    return trix, matrix