# AlgTradeTest
一个基于numpy pandas ta-lib 的简易交易策略测试程序，使用Cython把主逻辑加速了，可以setup用编译；
封装了okcoin的API，更多API接口和功能会持续更新。

## Requirements
* numpy
* pandas
* ta-lib
* Cython(非必需)

## How to Use
打开settings 按照需求修改参数
命令行运行 python run.py

### eva & eva_weight
* eva 是 技术分析指标的符号函数映射值之和，用于确定程序行为。</br>
```python
    k, d = trade.SRSI(df0, *argSRSI)
    hist, macd, sign  = trade.MACD(df0, *argMACD)
    eva = np.nan_to_num(np.sign(k-d) + np.sign(sign)*2)
```

* eva_weight 是 eva符号改变时总仓位变化的百分比。</br>
```python
    #columns -3 -1 1 3
    #row -3 -1 1 3
    eva_weight={
        -3:[-0.2, -0.05, -0.3, -0.7],
        -1:[-0.1, -0.05, -0.2, -0.5],
        1:[0.5, 0.2, 0.05, 0.1],
        3:[0.7, 0.3, 0.05, 0.2],
        'max':1,
        'stop_earn':0.1,
        'stop_loss':-0.05,
        'min_earn':0.01
    }
```
