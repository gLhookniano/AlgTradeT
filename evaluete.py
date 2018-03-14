#encoding: utf-8
import numpy as np
import trade
import talib as ta

import settings
MAKER_COST=settings.evaluete["maker_cost"]
TAKER_COST=settings.evaluete["taker_cost"]
IMPACT=settings.evaluete["impact"]
LEVER=settings.evaluete["lever"]

#SLtype 0 : short
#       1 : long
#flag_rate 0 : return rate
#          1 : return price
#flag_mt 0 : taker
#        1 : maker
def get_profit(open_price, close_price, SLtype, maker_cost=MAKER_COST, taker_cost=TAKER_COST, impact=IMPACT, lever=LEVER, flag_rate=1, flag_mt=0):
    cost, profit, open_cost=0,0,0

    if not flag_mt:
        cost=taker_cost
    else:
        cost=maker_cost
    if SLtype:
        profit = close_price*(1-impact)*(1-cost) - open_price*(1+impact)*(1+cost)
        if not flag_rate:
            return profit*lever
        if flag_rate:
            open_cost = (open_price)*(1+impact)*(1+cost)
            return profit/open_cost*lever
    if not SLtype:
        profit = open_price*(1-impact)*(1-cost) - close_price*(1+impact)*(1+cost)
        if not flag_rate:
            return profit*lever
        if flag_rate:
            open_cost = (open_price)*(1-impact)*(1+cost)
            return profit/open_cost*lever

    
def get_pure_profit(prof_rate, init_position=1.0):
    pure_profit=[]
    i=0
    for i in prof_rate:
        init_position*=(1+i)
        pure_profit.append(init_position)
    return pure_profit

    
def get_benchmark_profit(source, source_shift_1):
    change, bprofit=[],[]
    change = np.nan_to_num(source - source_shift_1)
    bprofit = change/source
    return bprofit

    
def get_beta(tatic_profit, benchmark_profit):
    m_cov=0
    cov, var, beta=0,0,0
    m_cov = np.cov((tatic_profit, benchmark_profit))
    cov = m_cov[0][1]
    var = np.var(benchmark_profit)
    beta = cov/var
    return beta


def get_alpha(tatic_profit,  benchmark_profit, riskfree_profit=0.01):
    beta, alpha=0,0
    beta = get_beta(tatic_profit, benchmark_profit)
    alpha = np.mean(tatic_profit - (riskfree_profit + beta*(benchmark_profit-riskfree_profit)))
    return alpha


def get_maxDrawDown(pure_profit, window=90):
    l_maxDD = []
    min, max, maxDD =0,0,0
    i=0
    for i in range(len(pure_profit)-window):
        min = np.min(pure_profit[i:i+window])
        max = np.max(pure_profit[i:i+window])
        l_maxDD.append(1-min/max)
    maxDD = np.max(l_maxDD)
    return maxDD

    
def get_sharpe(tatic_profit,  riskfree_profit):
    tf_mean, t_vol, sharpe = 0,0,0
    tf_mean = np.mean(tatic_profit - riskfree_profit)
    t_vol = np.std(tatic_profit)
    sharpe = tf_mean/t_vol
    return sharpe


def get_sortino(tatic_profit, riskfree_profit):
    tf_mean, t_vol_dw, sortino=0,0,0
    tf_mean = np.mean(tatic_profit - riskfree_profit)
    t_vol_dw = np.std(tatic_profit[tatic_profit<0])
    sortino = tf_mean/t_vol_dw
    return sortino


def get_winLose_analysis(l_profit):
    trade_time = len(l_profit)
    win_time = len(l_profit[l_profit>0])
    lose_time = len(l_profit[l_profit<0])
    victories = win_time/trade_time
    winEarn_rate = np.mean(l_profit[l_profit>0])
    loseLoss_rate = np.mean(l_profit[l_profit<0])
    odds = np.abs(winEarn_rate/loseLoss_rate)
    max_win_time=0
    max_lose_time=0
    slp=np.sign(l_profit)
    j=1
    k=1
    for i in range(1, len(slp)):
        if slp[i]>0 and slp[i-1]>0:
            j+=1
        if slp[i]<0 and slp[i-1]>0:
            if j>max_win_time:
                max_win_time=j+1
            j=1
        if slp[i]<0 and slp[i-1]<0:
            k+=1
        if slp[i]>0 and slp[i-1]<0:
            if k>max_lose_time:
                max_lose_time=k+1
            k=1
    return [trade_time, win_time, lose_time, victories, winEarn_rate, loseLoss_rate, odds, max_win_time, max_lose_time]


#flag_rtn 0: print all detail
#         1: return ditail
def get_backtest(profit, benchmark_profit, riskfree_profit=0.01, drawdown_window=90, flag_rtn=0):
    profit = np.array(profit)
    benchmark_profit = np.array(benchmark_profit)

    positive_profit = np.sum(profit[profit>0])
    negetive_profit = np.sum(profit[profit<0])
    pure_profit = get_pure_profit(profit)
    maxDrawDown = get_maxDrawDown(pure_profit, window=drawdown_window)
    beta = get_beta(profit, benchmark_profit)
    alpha = get_alpha(profit, benchmark_profit, riskfree_profit)
    sharpe = get_sharpe(profit,  riskfree_profit)
    sortino = get_sortino(profit, riskfree_profit)
    trade_time, win_time, lose_time, victories, winEarn_rate, loseLoss_rate, odds, max_win_time, max_lose_time = get_winLose_analysis(profit)
    if not flag_rtn:#print detail
        print('total positive profit:', positive_profit)
        print('total negetive profit:', negetive_profit)
        print('max Profit:           ', np.max(profit))
        print('min Profit:           ', np.min(profit))
        print('mean Profit:          ', np.mean(profit))
        print('var profit:           ', np.var(profit))
        print('pure profit max:      ', np.max(pure_profit))
        print('pure profit volatility:', np.var(pure_profit))
        print('maxDrawDown:          ', maxDrawDown)
        print('beta:                 ', beta)
        print('alpha:                ', alpha)
        print('sharpe:               ', sharpe)
        print('sortino:              ', sortino)
        print('total trade time:     ', trade_time)
        print('total win time:       ', win_time)
        print('total lose time:      ', lose_time)
        print('max continue win time:', max_win_time)
        print('max continue lose time:', max_lose_time)
        print('victories:            ', victories)
        print('win earn rate:        ', winEarn_rate)
        print('lose loss rate:       ', loseLoss_rate)
        print('odds:                 ', odds)
    else:
        return [positive_profit, negetive_profit, pure_profit, beta, alpha, maxDrawDown, sharpe, sortino, trade_time, win_time, lose_time, victories, winEarn_rate, loseLoss_rate, odds, max_win_time, max_lose_time]
    
    
def get_evaw_col(eva_weight):
    i,j=0,0
    evaw_col={}
    for i in eva_weight.keys():
        evaw_col[i]=j
        j+=1
    return evaw_col

def get_pst_ctl(now_eva, pre_eva, eva_weight, evaw_col):
    i,j=int(now_eva),int(pre_eva)
    pc=0
    
    if i in eva_weight.keys() and j in eva_weight.keys():#evaw_has_key
        pc = eva_weight[i][evaw_col[j]]
    else:#evaw_not_has_key
        pc=0
    return pc
    
'''
eva_weight={
-3:[-0.15, -0.05, -0.1, -0.3, -0.7],
-1:-0.05,
0:0,
1:0.05,
3:[0.7, 0.3, 0.1, 0.05, 0.15],
'max':1,
'stop_earn':0.1,
'stop_loss':-0.05,
'min_earn':0.001
}
'''
#flag_el_max 0: disable stop_earn and stop_loss
#            1: enable stop_earn and stop_loss
#flag_side_pst 0: use one-side trade
#              1: multi-side trade
#flag_e_min 0: limit min earn and ignore that trade
#           1: no limit
def tatic_eva(source,  eva, eva_weight=None, max_pst_num=1,  flag_el_max=False, flag_side_pst=False, flag_e_min=False):
    i,j,k,leva=0,0,0,len(eva)
    e_max=eva_weight.pop('stop_earn')#get_value and del_object_item
    l_max=eva_weight.pop('stop_loss')
    pc_max=eva_weight.pop('max')
    e_min=eva_weight.pop('min_earn')
    evaw_col = get_evaw_col(eva_weight)
    profit, i_pc, j_pc, now_price=0,0,0,0
    l_profit=[]
    pc_profit=[]
    
    for i in range(leva):
        if k>max_pst_num-1 and flag_side_pst:
            k-=1
            continue
        
        if i==0 or i==leva-1:#ig_first_eva || ig_end_eva
            l_profit.append(0)
            continue
        i_pc=0
        now_price=source[i]
        pc_profit=[]
        if eva[i]>0:
            for j in range(i+1, leva):
                k+=1
                if j==leva-1:#is_end_eva
                    l_profit.append(0)
                    continue
                j_pc = get_pst_ctl(eva[j], eva[j-1], eva_weight, evaw_col)#get position control value
                profit = get_profit(now_price, source[j], 1, flag_rate=True)
                
                if j_pc==-1 or j_pc+i_pc<0 or j==leva-1 or (profit>e_max or profit<l_max and flag_el_max):#full_out_long || L_min_position || end_bar || stop_earn || stop_loss
                    pc_profit.append(profit*i_pc)
                    l_profit.append(np.sum(pc_profit))
                    break
                
                elif j_pc==1 or j_pc+i_pc>=pc_max:#full_in_long || EB_max_position
                    j_pc=pc_max-i_pc
                    now_price = (i_pc*now_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc=pc_max
                
                elif j_pc<0:#out_long
                    if profit>0 and profit<=e_min and flag_e_min:#ig_earn_less
                        continue
                    pc_profit.append(profit*-j_pc)
                    i_pc+=j_pc
                    
                elif j_pc>0:#in_long
                    now_price = (i_pc*now_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc+=j_pc

        elif eva[i]<0:
            for j in range(i+1, leva):
                k+=1
                if j==leva-1:#is_end_eva
                    l_profit.append(0)
                    continue
                j_pc = get_pst_ctl(eva[j], eva[j-1], eva_weight, evaw_col)
                profit = get_profit(now_price, source[j], 0, flag_rate=True)
                
                if j_pc==1 or j_pc+i_pc>0 or j==leva-1 or (profit>e_max or profit<l_max and flag_el_max):#full_out_short || L_min_position || end_bar || stop_earn || stop_loss
                    pc_profit.append(profit*-i_pc)
                    l_profit.append(np.sum(pc_profit))
                    break
                    
                elif j_pc==-1 or -(j_pc+i_pc)>=pc_max:#full_in_short || EB_max_position
                    j_pc=-pc_max-i_pc
                    now_price = (i_pc*now_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc=-pc_max
                
                elif j_pc>0:#out_short
                    if profit>0 and profit<=e_min and flag_e_min:#ig_earn_less
                        continue
                    pc_profit.append(profit*j_pc)
                    i_pc+=j_pc
                    
                elif j_pc<0:#in_short
                    now_price = (i_pc*now_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc+=j_pc
        else:
            l_profit.append(0)
    return np.array(l_profit)