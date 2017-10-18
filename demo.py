# coding: utf-8

import gfunddataprocessor.gfunddataprocessor as gfd
import argparse
import copy
import pandas as pd
import math
import numpy as np

LONG = 1
SHORT = -1
CLOSE = 0

OLONG = 2
OSHORT = -2

TINY = 1e-4
NTINY = -1e-4

UP = 1
DOWN = -1


def enterobservation(tickbuff, malist, args):
    bandwidth = 20
    oblen = 60
    if len(tickbuff) < args.barsize*args.ma+oblen:
        return 0
    if tickbuff[-1] > malist[-1]+bandwidth:
        return LONG
    if tickbuff[-1] < malist[-1]-bandwidth:
        return SHORT
    return 0


def earn(price1, price2, dire):
    res = price1-price2 if dire > 0 else price2-price1
    return res


def leaveobservation(dire, tickbuff, tradebuff, args):
    bestprice = max(tradebuff) if dire == LONG else min(tradebuff)
    earnprice = earn(bestprice, tradebuff[0], dire)
    # if earnprice > 10*args.tickprice:
    waitlen = 600
    # if len(tradebuff) > 600 and \
    #         earn(tickbuff[-600], tradebuff[0], dire) <= -10.0 \
    #         and sum([e > tickbuff[-600] for e in tickbuff[-120:]]) == 0:
    #     return True
    # if earnprice > 10*args.tickprice and \
    #         earn(tickbuff[-1], tradebuff[0], dire) <= earnprice/4.0*3.0:
    #     return True

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category")
    parser.add_argument(
        "--start")
    parser.add_argument(
        "--end")
    parser.add_argument(
        "--tickprice",
        type=float)
    parser.add_argument(
        "--barsize",
        type=int)
    parser.add_argument(
        "--ma",
        type=int)
    parser.add_argument(
        "--extra",
        type=int,
        default=100)
    parser.add_argument(
        "--commision",
        type=float)
    args = parser.parse_args()
    if args.end is None:
        args.end = args.start

    datas = gfd.get_day_tick(args.category, (args.start, args.end),
                             data_source=gfd.YINHE, merge=False)

    datas = pd.concat(datas)
    profit = 0
    bestprice_ = None
    price_ = dire_ = None
    barlist = []        # 记录分钟级别bar的开高低收
    bardraw = []       # 记录分钟级别bar的收盘价
    bardraw1 = []
    bardraw2 = []
    tradebuff = []      # 记录开仓后所有的tick数据
    tickbuff = []       # 记录所有的tick价位
    tradetime = 0
    flagenter = flaglea = None
    needlen = args.barsize*args.ma + args.extra
    for data in [datas]:
        longmarks = []      # 记录做多时点
        shortmarks = []     # 记录做空时点
        closemarks = []     # 记录平仓时点
        daytime = 0         # 记录每天的交易次数

        # 输出策略日期
        # days = data.index[-1].strftime("%Y%m%d")
        # print "day: ", days
        # 记录一天数据长度， 最后半小时， 不再开仓
        daylen = len(data)
        lastenter = None
        reverse = False
        tickbuff = []
        for i in range(len(data)):
            item = data.iloc[i]
            tickbuff.append(item.price)
            nowprice = item.price
            if len(tickbuff) > needlen:
                del tickbuff[0]
            if len(tickbuff) < needlen:
                continue
            manow = sum([tickbuff[e] for e in
                         range(-1, -args.barsize*args.ma, -args.barsize)]) \
                / float(args.ma)
            barlist.append(manow)
            bardraw.append((i, manow))
            flagenter = enterobservation(tickbuff, barlist, args)
            if dire_ is None and i < len(data)-600:
                if flagenter == LONG:
                    price_ = item.price
                    bestprice_ = item.price
                    dire_ = LONG
                    longmarks.append((i, price_))
                    tradebuff.append(item.price)
                    daytime += 1
                    continue
                elif flagenter == SHORT:
                    price_ = item.price
                    bestprice_ = item.price
                    dire_ = SHORT
                    shortmarks.append((i, price_))
                    tradebuff.append(item.price)
                    daytime += 1
                    continue
                else:
                    continue

            # 已有持仓后，不在根据反方向信号离场， 而是根据leave逻辑
            # 时刻准备离场
            if dire_ is not None:
                tradebuff.append(item.price)
                flaglea = leaveobservation(dire_,
                                           tickbuff, tradebuff, args)
                if dire_ == LONG and (flagenter == SHORT or flaglea or i >= len(data)-600):
                # if dire_ == LONG and (flaglea or i >= len(data)-600):
                    oneprofit = item.bid_price1-price_
                    profit += oneprofit
                    print "profit: ", oneprofit, profit
                    price_ = item.bid_price1
                    dire_ = None
                    closemarks.append((i, price_))
                    tradebuff = []
                    tradetime += 1
                    continue
                if dire_ == SHORT and (flagenter == LONG or flaglea or i >= len(data)-600):
                # if dire_ == SHORT and (flaglea or i >= len(data)-600):
                    oneprofit = price_-item.ask_price1
                    profit += oneprofit
                    print "profit: ", oneprofit, profit
                    price_ = item.ask_price1
                    dire_ = None
                    closemarks.append((i, price_))
                    tradebuff = []
                    tradetime += 1
                    continue

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(200, 20))
        plt.plot(data.price.as_matrix())
        plt.plot([e[0] for e in longmarks], [p[1] for p in longmarks], 'r^')
        plt.plot([e[0] for e in shortmarks], [p[1] for p in shortmarks], 'yv')
        plt.plot([e[0] for e in closemarks], [p[1] for p in closemarks], 'ko')
        plt.plot([e[0] for e in bardraw], [p[1] for p in bardraw], 'g-')
        days = "glory"
        plt.savefig('mark_'+str(days)+'.png')
        plt.close()

        print "day trade time: ", daytime
    print "total tradetime: ", tradetime
