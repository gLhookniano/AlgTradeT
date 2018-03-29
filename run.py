# coding:utf-8
import glob
import numpy as np
import pandas as pd
import pstats, cProfile #profiling
from copy import deepcopy #test

import trade
import evaluete
from settings import run
LOC_DATA=run["loc_get_data"]
eva3_weight=run["eva3_weight"]
eva_weight=run["eva_weight"]
RF_PROFIT=run["riskfreeProfit"]
DRAWDOWN_WINDOW=run["drawdown_window"]
RUN_TYPE=run["RUN_TYPE"]
argSRSI=run["argSRSI"]
argMACD=run["argMACD"]
argBB=run["argBB"]
argATR=run["argATR"]
argDC=run["argDC"]
argKD=run["argKD"]
argWR=run["argWR"]
argSAR=run["argSAR"]
argMATRIX=run["argMATRIX"]

def get_kline_data(s_coin, s_ftdate, s_freq, l_groupby=['time','close']):
    l = glob.glob(LOC_DATA+'*'+s_coin+'*'+s_ftdate+'*'+s_freq+'*')
    df0=pd.DataFrame()
        
    for i in l:
        df = pd.read_csv(i, index_col=0)
        df0=pd.concat([df0,df])
    df0.index=range(len(df0))
    df0.time.apply(lambda x: int(str(x)[:10]))
    df0 = df0.groupby(l_groupby).tail(1)
    df0.index = range(len(df0))
    return df0

def profiling():
    def x():
        cProfile.runctx("test()", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    x()

def eva3_SRSI_MACD_histSign(df0, *args):
    k, d = trade.SRSI(df0, *args[0])
    hist, macd, sign  = trade.MACD(df0, *args[1])
    eva = np.nan_to_num(np.sign(k-d) + np.sign(hist)*2 + np.sign(sign)*3)
    return eva

def eva_SRSI_BB(df0, *args):
    K, D = trade.SRSI(df0, *args[0])
    u, m, d = trade.BB(df0, *args[1])
    
    uu = np.sign(u - np.array(df0.highest))
    uu[uu!=1]=0
    dd = np.sign(np.array(df0.lowest) - d)
    dd[dd!=-1]=0
    
    uu_shift = np.append(0, uu[:-1])
    uuu = uu - uu_shift
    uuu[uuu!=1]=0
    
    dd_shift = np.append(0, dd[:-1])
    ddd = dd - dd_shift
    ddd[ddd!=-1]=0
    
    eva = np.nan_to_num(np.sign(K-D))
    #eva = np.nan_to_num(np.sign(list(range(len(u))))*-1)
    
    eva4open = np.nan_to_num(uu+dd)
    return eva, eva4open

def eva_SRSI_ATR(df0, *args):
    atrU = args[1][1]
    atrD = args[1][2]
        
    k, d = trade.SRSI(df0, *args[0])
    atr = trade.ATR(df0, args[1][0])
    atr[atr>=atrU]=1
    atr[atr<=atrD]=-1
    atr[np.abs(atr)!=1]=0
    
    eva = np.nan_to_num(np.sign(k-d) + np.sign(atr)*2)
    return eva

def eva_SRSI_DC(df0, *args):
    K, D = trade.SRSI(df0, *args[0])
    u, d = trade.DC(df0, *args[1])
    
    uu = np.sign(np.array(df0.highest) - u)
    uu[uu<0]=0
    dd = np.sign(np.array(df0.lowest) - d)
    dd[dd>0]=0
    eva = np.nan_to_num(np.sign(K-D) + np.sign(uu+dd)*2)
    return eva

def eva_KD_WR(df0, *args):
    wrU=args[1][1]
    wrD=args[1][2]

    K, D = trade.KD(df0, *args[0])
    wr = trade.WR(df0, args[1][0])
    wr[wr>=wrU]=1
    wr[wr<=wrD]=-1
    wr[np.abs(wr)!=1]=0
    
    eva = np.nan_to_num(np.sign(K-D) + np.sign(wr)*2)
    return eva
    
def eva3_SAR_MACD_histSign(df0, *args):
    sar = trade.SAR(df0, *args[0])
    hist, macd, sign  = trade.MACD(df0, *args[1])
    sar_shift_1 = np.append(0, sar[:-1])
    
    eva = np.nan_to_num(np.sign(sar - sar_shift_1) + np.sign(hist)*2 + np.sign(sign)*3)
    return eva
    
def eva_KD_MATRIX(df0, *args):
    K, D = trade.KD(df0, *args[0])
    trix, matrix = trade.MATRIX(df0, *args[1])
    
    eva = np.nan_to_num(np.sign(K-D) + np.sign(matrix - trix)*2)
    return eva
    
def test(flag_type="SRSI_MACD_histSign"):
    df4 = get_kline_data('btc','next','1min',['time', 'close'])
    source = np.array(df4.close)
    
    if flag_type=="SRSI_MACD_histSign":
        eva = eva3_SRSI_MACD_histSign(df4, argSRSI, argMACD)
        use_eva_weight = deepcopy(eva3_weight)
    if flag_type=="SRSI_BB":
        eva, eva4open = eva_SRSI_BB(df4, argSRSI, argBB)
        use_eva_weight = deepcopy(eva_weight)
    if flag_type=="SRSI_ATR":
        eva = eva_SRSI_ATR(df4, argSRSI, argATR)
        use_eva_weight = deepcopy(eva_weight)
    if flag_type=="SRSI_DC":
        eva = eva_SRSI_DC(df4, argSRSI, argDC)
        use_eva_weight = deepcopy(eva_weight)
    if flag_type=="KD_WR":
        eva = eva_KD_WR(df4, argKD, argWR)
        use_eva_weight = deepcopy(eva_weight)
    if flag_type=="SAR_MACD_histSign":
        eva = eva3_SAR_MACD_histSign(df4, argSAR, argMACD)
        use_eva_weight = deepcopy(eva3_weight)
    if flag_type=="KD_MATRIX":
        eva =  eva_KD_MATRIX(df4, argKD, argMATRIX)
        use_eva_weight = deepcopy(eva_weight)
        
    #profit = evaluete.tatic_eva(source, eva=eva, eva_weight=use_eva_weight)
    profit = evaluete.tatic_eva(df4, eva, eva4open, use_eva_weight, 1, [0]).do()
    
    pure_profit = evaluete.get_pure_profit(profit)
    ben_profit = evaluete.get_benchmark_profit(source, np.array(df4.close.shift(1)))
    evaluete.get_backtest(profit, ben_profit, RF_PROFIT, DRAWDOWN_WINDOW)
        
    dpv=pd.DataFrame([pure_profit], columns=range(len(pure_profit)), index=['pure_profit']).T
    dp=pd.DataFrame([profit], columns=range(len(profit)), index=['profit']).T
    de=pd.DataFrame([eva], columns=range(len(eva)), index=['eva']).T
    df = pd.concat([df4.highest, df4.lowest, df4.close, de, dp, dpv], axis=1)
    print(df)
        
if __name__ == '__main__':
    import time
    t0 = time.time()
    test(RUN_TYPE)
    print(time.time()-t0)
    