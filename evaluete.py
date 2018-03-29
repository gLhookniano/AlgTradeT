# coding: utf-8
import numpy as np
import talib as ta

from settings import evaluete
MAKER_COST = evaluete["maker_cost"]
TAKER_COST = evaluete["taker_cost"]
IMPACT = evaluete["impact"]
SLIDE = evaluete["slide"]
LEVER = evaluete["lever"]
MAX_POSITION = evaluete["max_position"]
STOP_EARN = evaluete["stop_earn"]
STOP_LOSS = evaluete["stop_loss"]


class tatic_eva(object):
    """
    tatic_eva(self, df0, eva, eva4open, eva_weight=None, flag_riskcontrol=False,  flag_riskcontrol_type=0)
    
    flag_riskcontrol 0: disable stop_earn and stop_loss
                     1: enable stop_earn and stop_loss
    flag_riskcontrol_type 0: use const percent
                          1: use atrN
                          2: use sar sign
                          3: use sar follow
                
    return: <float>list  (which is profit list)
    """
    def __init__(self, df0, eva, eva4open, eva_weight=None, flag_riskcontrol=False,  flag_riskcontrol_type=0):
        self.l_profit=[]
        self.eva=eva
        self.eva4open = eva4open
        self.eva_weight = eva_weight
        self.source = np.array(df0.close)
        self.high = np.array(df0.highest)
        self.low = np.array(df0.lowest)
        
        self.flag_riskcontrol = flag_riskcontrol
        self.flag_riskcontrol_type = flag_riskcontrol_type
        self.riskControl_profit = []

    def get_pst_ctl(self, now_eva, pre_eva, eva_weight, evaw_col):
        i,j,pc=int(now_eva),int(pre_eva),0
        if i in eva_weight.keys() and j in eva_weight.keys():#evaw_has_key
            pc = eva_weight[i][evaw_col[j]]
        else:#evaw_not_has_key
            pc=0
        return pc

    def get_evaw_col(self, eva_weight):
        """
        eva_weight={#-3     -1     1    3
                -3:[-0.2, -0.05, -0.3, -0.7],
                -1:[-0.1, -0.05, -0.2, -0.5],
                1:[0.5, 0.2, 0.05, 0.1],
                3:[0.7, 0.3, 0.05, 0.2],
                }
        """
        i,j=0,0
        evaw_col={}
        for i in eva_weight.keys():
            evaw_col[i]=j
            j+=1
        return evaw_col
        
    def get_open_price(self, time, SLtype):
        """
        get_open_price(self, time)
        
        return: <float>
        """
        if SLtype==1:
            price = self.source[time]*(1+SLIDE)
        if SLtype==0:
            price = self.source[time]*(1-SLIDE)
        return price
        
    def get_close_price(self, time):
        """
        get_close_price(self, time)
        
        return: <float>
        """
        if SLtype==1:
            price = self.source[time]*(1-SLIDE)
        if SLtype==0:
            price = self.source[time]*(1+SLIDE)
        return price
        
    def get_open_sign(self, time):
        """
        get_open_sign(self, time)
             
        return 0: goto short condition
               1: goto long condition
               -1: goto zero condition
        """
        if self.eva4open[time]==-1:
            return 0
        if self.eva4open[time]==1:
            return 1
        else:
            return -1
        
    def do(self):
        return self.loop_time()
        
    def loop_time(self):
        evaw_col = self.get_evaw_col(self.eva_weight)
        leva = len(self.eva)
    
        for i in range(leva):
            if i==0 or i==leva-1:#ig_first_time || ig_end_time
                self.l_profit.append(0)
                continue
            
            open_sign = self.get_open_sign(i)
            if open_sign==1:
                eva_profit = self.loop_eva_long(i, leva, evaw_col)
                self.l_profit.append(np.sum(eva_profit))
            elif open_sign==0:
                eva_profit = self.loop_eva_short(i, leva, evaw_col)
                self.l_profit.append(np.sum(eva_profit))
            elif open_sign==-1:
                self.l_profit.append(0)
        return self.l_profit
        
    def loop_eva_long(self, time, leva, evaw_col):
        """
        loop_eva_long(self, time, leva, evaw_col)
        time : now time
        """
        i_pc = 0
        eva_profit = []
        open_price = self.get_open_price(time)
        
        for j in range(time+1, leva):
            j_pc = self.get_pst_ctl(self.eva[j], self.eva[j-1], self.eva_weight, evaw_col)#get position control value
            profit = get_profit(open_price, self.get_close_price(j), 1, flag_rate=True)
                
            if j_pc==-1 or j_pc+i_pc<0 or j==leva-1 or (self.riskControl(profit, open_price, time, j, 1) and self.flag_riskcontrol):#full_out_long || L_min_position || end_bar || (stop_earn || stop_loss && flag_stop)
                eva_profit.append(profit*i_pc)
                break
                
            elif j_pc==1 or j_pc+i_pc>=MAX_POSITION:#full_in_long || EB_max_position
                j_pc=MAX_POSITION-i_pc
                open_price = (i_pc*open_price + j_pc*self.get_open_price(j)) / (i_pc+j_pc)
                i_pc=MAX_POSITION
                
            elif j_pc<0:#out_long
                eva_profit.append(profit*-j_pc)
                i_pc+=j_pc
                    
            elif j_pc>0:#in_long
                open_price = (i_pc*open_price + j_pc*self.get_open_price(j)) / (i_pc+j_pc)
                i_pc+=j_pc
        return eva_profit
        
    def loop_eva_short(self, time, leva, evaw_col):
        """
        loop_eva_short(self, time, leva, evaw_col)
        time : now time
        """
        i_pc = 0
        eva_profit = []
        open_price = self.get_open_price(time)
        
        for j in range(time+1, leva):
            j_pc = self.get_pst_ctl(self.eva[j], self.eva[j-1], self.eva_weight, evaw_col)
            profit = get_profit(open_price, self.get_close_price(j), 0, flag_rate=True)
            
            if j_pc==1 or j_pc+i_pc>0 or j==leva-1 or (self.riskControl(profit, open_price, time, j, 0) and self.flag_riskcontrol):#full_out_short || L_min_position || end_bar || (stop_earn || stop_loss && flag_stop)
                eva_profit.append(profit*-i_pc)
                break
                
            elif j_pc==-1 or -(j_pc+i_pc)>=MAX_POSITION:#full_in_short || EB_max_position
                j_pc=-MAX_POSITION-i_pc
                open_price = (i_pc*open_price + j_pc*self.get_open_price(j)) / (i_pc+j_pc)
                i_pc=-MAX_POSITION
                
            elif j_pc>0:#out_short
                eva_profit.append(profit*j_pc)
                i_pc+=j_pc
                
            elif j_pc<0:#in_short
                open_price = (i_pc*open_price + j_pc*self.get_open_price(j)) / (i_pc+j_pc)
                i_pc+=j_pc
        return eva_profit
        
    def riskControl(self, profit, open_price, open_time, time, SLtype):
        """
        riskControl(self, profit, open_price, open_time, time, type)
        
        self.flag_riskcontrol_type:
            0 : percent
            1 : atrN
            2 : sar
            3 : sarx
        """
        l=[]
        if 0 in self.flag_riskcontrol_type:
            l.append(self.riskControl_percent(profit))
        if 1 in self.flag_riskcontrol_type:
            l.append(self.riskControl_atrN(open_price, time, SLtype))
        if 2 in self.flag_riskcontrol_type:
            l.append(self.riskControl_sar(open_price, time, SLtype))
        if 3 in self.flag_riskcontrol_type:
            l.append(self.riskControl_sarx(open_price, open_time, time, SLtype))
        if 4 in self.flag_riskcontrol_type:
            l.append(self.riskControl_drawdown(profit, open_time, time))
        
        if True in l:
            return True
        else:
            return False
            
    def riskControl_percent(self, profit):
        """
        use simple const percent
        """
        if profit >= STOP_EARN or profit <= STOP_LOSS:
            return True
        else:
            return False
    def riskControl_atrN(self, open_price, time, SLtype, window=14, n_earn=2.5, n_loss=.5):
        """
        riskControl_atrN(profit, open_price, window=14, n_earn=2.5, n_loss=.5)
        
        
        if n_earn*now_atr < now_close-open then stop_earn; default 2.5
        if n_loss*now_atr > open-now_close then stop_loss; default 0.5

        return: float
        """
        if time<window:
            return False
        atr = ta.ATR(self.high[time-window:time+1], self.low[time-window:time+1], self.source[time-window:time+1], timeperiod=window)
        if SLtype:
            if atr[-1]*n_earn < (self.source[time] - open_price) or atr[-1]*n_loss > (open_price - self.source[time]):
                return True
            else:
                return False
        elif not SLtype:
            if atr[-1]*n_earn < (open_price - self.source[time]) or atr[-1]*n_loss > (self.source[time] - open_price):
                return True
            else:
                return False
    def riskControl_sar(self, open_price, time, SLtype, AF_increse=0.01, AF_max=0.1):
        """
        riskControl_sar(self, open_price, time, type, AF_increse=0.01, AF_max=0.1)
        
        type 0 : short
             1 : long
             
        return: float
        """
        sar = ta.SAR(self.high[:time+1], self.low[:time+1], acceleration=AF_increse, maximum=AF_max)
        if SLtype==1:
            if sar[-1]>=0:
                return False
            elif sar[-1]<0:
                return True
        if SLtype==0:
            if sar[-1]<=0:
                return False
            elif sar[-1]>0:
                return True
        return False
    def riskControl_sarx(self, open_price, open_time, time, SLtype, AF_increse=0.01, AF_max=0.1):
        """
        riskControl_sarx(self, open_price, open_time, time, type, AF_increse=0.02, AF_max=0.2)
        
        type 0 : short
             1 : long
             
        return: float
        """
        if SLtype==1:
            sarx = ta.SAREXT(self.high[open_time:time+1], self.low[open_time:time+1], startvalue=open_price, offsetonreverse=0, accelerationinitlong=AF_increse, accelerationlong=AF_increse, accelerationmaxlong=AF_max, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
            if sarx[-1]>=0:
                return False
            elif sarx[-1]<0:
                return True
        if SLtype==0:
            sarx = ta.SAREXT(self.high[:time+1], self.low[:time+1], startvalue=open_price, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=AF_max, accelerationinitshort=AF_increse, accelerationshort=AF_increse, accelerationmaxshort=AF_max)
            if sarx[-1]<=0:
                return False
            elif sarx[-1]>0:
                return True
        return False
    def riskControl_drawdown(self, profit, open_time, time, n_drawdown=.5):
        if open_time == time+1:#ig_first_time
            self.riskControl_profit=[]
            return False
        self.riskControl_profit.append(profit)
        if self.riskControl_profit[-1]/np.max(self.riskControl_profit)<n_drawdown:
            return True
        return False
def get_profit(open_price, close_price, SLtype, maker_cost=MAKER_COST, taker_cost=TAKER_COST, impact=IMPACT, lever=LEVER, flag_rate=1, flag_mt=0):
    """
    get_profit(open_price, close_price, SLtype, maker_cost=MAKER_COST, taker_cost=TAKER_COST, impact=IMPACT, lever=LEVER, flag_rate=1, flag_mt=0)
    
    SLtype 0 : short
            1 : long
    flag_rate 0 : return rate
            1 : return price
    flag_mt 0 : taker
            1 : maker
            
    return: profit
    """
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
            open_cost = (open_price)*(1+impact)*(1+cost)
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
    cov, m_cov, var, beta=0,0,0,0
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
    STOP_LOSSDD = []
    min, max, maxDD =0,0,0
    i=0
    for i in range(len(pure_profit)-window):
        min = np.min(pure_profit[i:i+window])
        max = np.max(pure_profit[i:i+window])
        STOP_LOSSDD.append(1-min/max)
    maxDD = np.max(STOP_LOSSDD)
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


def get_backtest(profit, benchmark_profit, riskfree_profit=0.01, drawdown_window=90, flag_rtn=0):
    """
    flag_rtn 0: print all detail
            1: return ditail
    """
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
