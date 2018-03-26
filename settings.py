# coding:utf-8

okcoinAPI=dict(
api_Key='XXX',
secret_Key = "XXX",
url = "wss://real.okex.com:10440/websocket/okexapi",
url_spot = "wss://real.okex.com:10441/websocket/okexapi",
loc_dump_data = r"./DATA/",
)

evaluete=dict(
maker_cost=0.0003,
taker_cost=0.0005,
impact=0.0001,
lever=10,
max_position=1,
stop_earn=0.05,
stop_loss=-0.05,
)

run=dict(
loc_get_data=r"./DATA/",
eva3_weight={
            -6:[-.2, -.15,  -.1, -.2, -.3, -.6, -.9],#-1-2-3
            -4:[-.1, -.05, -.05, -.15, -.2, -.3, -.5],#+1-2-3
            -2:[-.05, -.05,  -.05, -.1, -.1, -.3, -.5],#-1+2-3
            0:[.3, .2, .1, 0, -.1, -.2, -.3],#-1-2+3 and 1+2-3
            2:[.5, .3, .1, .1,  .05, .05, .05],#1-2+3
            4:[.5, .3, .2, .15, .05, .05, .1],#-1+2+3
            6:[.9, .6, .3, .2,  .1, .15, .2],#1+2+3
            },
eva_weight={
            -3:[-0.2, -0.05, -0.3, -0.7],
            -1:[-0.1, -0.05, -0.2, -0.5],
            1:[0.5, 0.2, 0.05, 0.1],
            3:[0.7, 0.3, 0.05, 0.2],
            },
riskfreeProfit=0.01,
drawdown_window=60,
RUN_TYPE="KD_MATRIX",#SRSI_MACD_histSign, SRSI_BB, SRSI_ATR*, SRSI_DC, KD_WR, SAR_MACD*, KD_MATRIX
argSRSI=(9,5,3),    #window, win_k, win_d
argMACD=(12,26,9),  #window_fast, window_slow, window_sign
argBB=(4,2,2),      #window, nbdevup, nbdevdn
argATR=(14,0.8,0.3),#window, atrUpLine, atrDownLine
argDC=(20,),        #window
argKD=(5,3,3),      #window_k_fast, window_k, window_d
argWR=(14,-20,-80), #window, wrUpLine, wrDownLine
argSAR=(0.02,0.2),  #AF_increase, AF_max
argMATRIX=(25,14),  #window_trix, win_matrix
)