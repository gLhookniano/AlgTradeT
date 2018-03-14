#encoding: utf-8
import cython
from cpython cimport array
import array
import numpy as np
cimport numpy as np
from talib import SAREXT

from settings import evaluete
MAKER_COST=evaluete["maker_cost"]
TAKER_COST=evaluete["taker_cost"]
IMPACT=evaluete["impact"]
LEVER=evaluete["lever"]

#SLtype 0 : short
#       1 : long
#flag_rate 0 : return rate
#          1 : return price
#flag_mt 0 : taker
#        1 : maker
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _get_profit(double open_price, double close_price, bint SLtype, double maker_cost=MAKER_COST, double taker_cost=TAKER_COST, double impact=IMPACT, int lever=LEVER, bint flag_rate=1, bint flag_mt=0) except -1:
    cdef:
        double cost, profit, open_cost

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
def get_profit(open_price, close_price, SLtype, maker_cost=0.0003, taker_cost=0.0005, impact=0.0001, lever=10, flag_rate=0, flag_mt=0):
    return _get_profit(open_price, close_price, SLtype, maker_cost, taker_cost, impact, lever, flag_rate, flag_mt)
   
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _get_pure_profit(double[:] prof_rate, double position=1.0):
    cdef:
        array.array pure_profit=array.array('d')
        double i=0
    for i in prof_rate:
        position*=(1.+i)
        pure_profit.append(position)
    return pure_profit
@cython.boundscheck(False)
@cython.wraparound(False)
def get_pure_profit(double[:] prof_rate, double init_position=1.0):
    return _get_pure_profit(prof_rate, init_position)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_benchmark_profit(source, source_shift_1):
    change, bprofit=[],[]
    change = np.nan_to_num(source - source_shift_1)
    bprofit = change/source
    return bprofit

@cython.boundscheck(False)
@cython.wraparound(False)
def get_beta(tatic_profit, benchmark_profit):
    m_cov=0
    cov, var, beta=0,0,0
    m_cov = np.cov((tatic_profit, benchmark_profit))
    cov = m_cov[0][1]
    var = np.var(benchmark_profit)
    beta = cov/var
    return beta

@cython.boundscheck(False)
@cython.wraparound(False)
def get_alpha(tatic_profit,  benchmark_profit, riskfree_profit=0.01):
    beta, alpha=0,0
    beta = get_beta(tatic_profit, benchmark_profit)
    alpha = np.mean(tatic_profit - (riskfree_profit + beta*(benchmark_profit-riskfree_profit)))
    return alpha

@cython.boundscheck(False)
@cython.wraparound(False)
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

@cython.boundscheck(False)
@cython.wraparound(False)
def get_sharpe(tatic_profit,  riskfree_profit):
    tf_mean, t_vol, sharpe = 0,0,0
    tf_mean = np.mean(tatic_profit - riskfree_profit)
    t_vol = np.std(tatic_profit)
    sharpe = tf_mean/t_vol
    return sharpe

@cython.boundscheck(False)
@cython.wraparound(False)
def get_sortino(tatic_profit, riskfree_profit):
    tf_mean, t_vol_dw, sortino=0,0,0
    tf_mean = np.mean(tatic_profit - riskfree_profit)
    t_vol_dw = np.std(tatic_profit[tatic_profit<0])
    sortino = tf_mean/t_vol_dw
    return sortino

@cython.boundscheck(False)
@cython.wraparound(False)
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
@cython.boundscheck(False)
@cython.wraparound(False)
def get_backtest(profit, benchmark_profit, riskfree_profit=0.01, drawdown_window=90, flag_rtn=0):
    profit = np.array(profit)
    benchmark_profit = np.array(benchmark_profit)

    positive_profit = np.sum(profit[profit>0])
    negetive_profit = np.sum(profit[profit<0])
    pure_profit = _get_pure_profit(profit)
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
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _get_evaw_col(object eva_weight):
    cdef:
        int i=0,j=0
    evaw_col={}
    for i in eva_weight.keys():
        evaw_col[i]=j
        j+=1
    return evaw_col
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _get_pst_ctl(double now_eva, double pre_eva, object eva_weight, object evaw_col):
    cdef:
        int i=<int>now_eva, j=<int>pre_eva
        double pc=0.
    if i in eva_weight.keys() and j in eva_weight.keys():#evaw_has_key
        pc = eva_weight[i][evaw_col[j]]
    else:#evaw_not_has_key
        pc=0.
    return pc
    
@cython.boundscheck(False)
@cython.wraparound(False)
def gen_test_evaw(test_step=0.05, eva_weight=None):
    if not eva_weight:
        eva_weight_str_end={
            -3:[(-.2,.05), (-.2,-.05), (-.5,-.2), (-.9,-.3)],
            -1:[(-.2,.05),    (-.1,0),   (-.3,0), (-.7,-.3)],
            1:[(.3,.7), (.1,.5),   (0,.3),  (0,0.3)],
            3:[(.3,.9), (.2,.5), (.05,.2), (.05,.2)],
            'max':[1],
            'stop_earn':[(0.05,0.5)],
            'stop_loss':[(-0.2,0)],
            'min_earn':[(0.05,0.2)],
            }
    else:
        eva_weight_str_end=eva_weight
    for row in eva_weight_str_end.keys():
        for col in range(len(eva_weight_str_end[row])):
            eva_weight_str_end[row][col] = [i for i in np.arange(eva_weight_str_end[row][col][0],eva_weight_str_end[row][col][1],test_step)].append(eva_weight_str_end[row][col][1])
    return eva_weight_str_end
    
'''
eva_weight={
-3:[-0.15, -0.05, -0.1, -0.3, -0.7],
-1:-0.05,
1:0.05,
3:[0.7, 0.3, 0.1, 0.05, 0.15],
'max':1,
'stop_earn':0.1,
'stop_loss':-0.05,
'min_earn':0.001
}
'''
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=1] _tatic_eva(double[:] source, double[:] eva, object eva_weight, int max_pst_num, bint flag_el_max, bint flag_side_pst, bint flag_e_min):
    cdef:
        int i=0,j=0,k=0,leva=len(eva)
        double e_max=eva_weight.pop('stop_earn')#get_value and del_object_item
        double l_max=eva_weight.pop('stop_loss')
        double pc_max=eva_weight.pop('max')
        double e_min=eva_weight.pop('min_earn')
        double profit=0, i_pc=0, j_pc=0, open_price=0
        object evaw_col=_get_evaw_col(eva_weight)
        array.array l_profit=array.array('d'), pc_profit=array.array('d')
        
    
    for i in range(leva):
        if k>max_pst_num-1 and flag_side_pst:#is_max_position_one_time
            k-=1
            continue
        i_pc=0
        open_price=source[i]
        pc_profit = array.array('d')
        if eva[i]>0:
            for j in range(i, leva):
                k+=1
                if j==i:#ig_first_eva
                    if j==leva-1:#ig_end_eva
                        l_profit.append(0)
                    continue
                j_pc = _get_pst_ctl(eva[j], eva[j-1], eva_weight, evaw_col)#get position control value
                profit = _get_profit(open_price, source[j], 1)
                
                if j_pc==-1 or j_pc+i_pc<0 or j==leva-1 or (profit>e_max or profit<l_max and flag_el_max):#full_out_long || L_min_position || end_bar || stop_earn || stop_loss
                    pc_profit.append(profit*i_pc)
                    l_profit.append(np.sum(pc_profit))
                    break
                
                elif j_pc==1 or j_pc+i_pc>=pc_max:#full_in_long || EB_max_position
                    j_pc=pc_max-i_pc
                    open_price = (i_pc*open_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc=pc_max
                
                elif j_pc<0:#out_long
                    if profit>0 and profit<=e_min and flag_e_min:#ig_earn_less
                        continue
                    pc_profit.append(profit*-j_pc)
                    i_pc+=j_pc
                    
                elif j_pc>0:#in_long
                    open_price = (i_pc*open_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc+=j_pc

        elif eva[i]<0:
            for j in range(i, leva):
                k+=1
                if j==i:#ig_first_eva
                    if j==leva-1:#ig_end_eva
                        l_profit.append(0)
                    continue
                j_pc = _get_pst_ctl(eva[j], eva[j-1], eva_weight, evaw_col)
                profit = _get_profit(open_price, source[j], 0)
                
                if j_pc==1 or j_pc+i_pc>0 or j==leva-1 or (profit>e_max or profit<l_max and flag_el_max):#full_out_short || L_min_position || end_bar || stop_earn || stop_loss
                    pc_profit.append(profit*-i_pc)
                    l_profit.append(np.sum(pc_profit))
                    break
                    
                elif j_pc==-1 or -(j_pc+i_pc)>=pc_max:#full_in_short || EB_max_position
                    j_pc=-pc_max-i_pc
                    open_price = (i_pc*open_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc=-pc_max
                
                elif j_pc>0:#out_short
                    if profit>0 and profit<=e_min and flag_e_min:#ig_earn_less
                        continue
                    pc_profit.append(profit*j_pc)
                    i_pc+=j_pc
                    
                elif j_pc<0:#in_short
                    open_price = (i_pc*open_price + j_pc*source[j]) / (i_pc+j_pc)
                    i_pc+=j_pc
        else:
            l_profit.append(0)
    return np.array(l_profit)
@cython.boundscheck(False)
@cython.wraparound(False)
def tatic_eva(double[:] source, double[:] eva, object eva_weight=None, int max_pst_num=1, bint flag_el_max=False, bint flag_side_pst=False, bint flag_e_min=False):
    return _tatic_eva(source, eva, eva_weight, max_pst_num,  flag_el_max, flag_side_pst, flag_e_min)

#tend 1: long
#     0: short
@cython.boundscheck(False)
@cython.wraparound(False)
cdef SARX(np.ndarray[np.double_t, ndim=1] highest, np.ndarray[np.double_t, ndim=1] lowest, double init_val, bint tend, int init_index=0, double rev_val_per=0.002, double AF_init=0.02, double AF_increse=0.02, double AF_max=0.2):
    sarx = []
    if tend:
        sarx = SAREXT(highest, lowest, startvalue=init_val, offsetonreverse=rev_val_per, accelerationinitlong=AF_init, accelerationlong=AF_increse, accelerationmaxlong=AF_max, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
        return sarx
    if not tend:
        sarx = SAREXT(highest, lowest, startvalue=init_val, offsetonreverse=rev_val_per, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=AF_init, accelerationshort=AF_increse, accelerationmaxshort=AF_max)
        return sarx