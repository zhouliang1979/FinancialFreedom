# coding: utf-8
"""
    这是一个简单的均线策略， 当行情突破 均线+bandwidth之后， 上突破做多， 下突破做空
    反向突破时则离场
    运行指令：python demo.py --category rb1801 --start 20170823 --end 20170901 --barsize 1800 --ma
    35 --bandwidth 20 --tickprice 1

    数据文件名必须是 合约名.csv, 比如rb1801.csv
"""

import argparse
import copy
import pandas as pd
import math
import numpy as np
import csv

LONG = 1
SHORT = -1
CLOSE = 0

OLONG = 2
OSHORT = -2

TINY = 1e-4
NTINY = -1e-4

UP = 1
DOWN = -1

def get_day_tick(contract, interval, merge=False):
    filename = contract + '.csv'
    content = csv.DictReader(open(filename, 'r'))
    fitcontent = []
    for line in content:
        TradingDay = line['TradingDay'].replace('-', '')
        if TradingDay >= interval[0] and TradingDay <= interval[1]:
            fitcontent.append(line)
        if TradingDay > interval[1]:
            break
    df = pd.DataFrame(fitcontent)
    if merge:
        return df
    else:
        days = df.TradingDay.unique()
        days.sort()
        dflist = []
        for day in days:
            singledf = df[df.TradingDay == day]
            dflist.append(pd.DataFrame(singledf))
        return dflist


def enterobservation(tickbuff, malist, args):
    bandwidth = args.bandwidth
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
    waitlen = 600

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
        "--bandwidth",
        type=int,
        default=20)
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

    datas = get_day_tick(args.category, (args.start, args.end))

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

        daylen = len(data)
        lastenter = None
        reverse = False
        tickbuff = []
        for i in range(len(data)):
            item = data.iloc[i]
            tickbuff.append(float(item.price))
            nowprice = float(item.price)
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
                    price_ = float(item.price)
                    bestprice_ = float(item.price)
                    dire_ = LONG
                    longmarks.append((i, price_))
                    tradebuff.append(price_)
                    daytime += 1
                    continue
                elif flagenter == SHORT:
                    price_ = float(item.price)
                    bestprice_ = float(item.price)
                    dire_ = SHORT
                    shortmarks.append((i, price_))
                    tradebuff.append(price_)
                    daytime += 1
                    continue
                else:
                    continue

            # 已有持仓后，根据leave逻辑或者反向入场信号离场
            if dire_ is not None:
                tradebuff.append(float(item.price))
                flaglea = leaveobservation(dire_,
                                           tickbuff, tradebuff, args)
                if dire_ == LONG and (flagenter == SHORT or flaglea or i >= len(data)-600):
                    oneprofit = float(item.bid_price1)-price_
                    profit += oneprofit
                    print "profit: ", oneprofit, profit
                    price_ = float(item.bid_price1)
                    dire_ = None
                    closemarks.append((i, price_))
                    tradebuff = []
                    tradetime += 1
                    continue
                if dire_ == SHORT and (flagenter == LONG or flaglea or i >= len(data)-600):
                    oneprofit = price_-float(item.ask_price1)
                    profit += oneprofit
                    print "profit: ", oneprofit, profit
                    price_ = float(item.ask_price1)
                    dire_ = None
                    closemarks.append((i, price_))
                    tradebuff = []
                    tradetime += 1
                    continue

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(200, 20))
        plt.plot([float(e) for e in data.price])
        plt.plot([e[0] for e in longmarks], [p[1] for p in longmarks], 'r^')
        plt.plot([e[0] for e in shortmarks], [p[1] for p in shortmarks], 'yv')
        plt.plot([e[0] for e in closemarks], [p[1] for p in closemarks], 'ko')
        plt.plot([e[0] for e in bardraw], [p[1] for p in bardraw], 'g-')
        plt.plot([e[0] for e in bardraw], [p[1]-args.bandwidth for p in bardraw], 'g-')
        plt.plot([e[0] for e in bardraw], [p[1]+args.bandwidth for p in bardraw], 'g-')
        days = "glory"
        plt.savefig('mark_'+str(days)+'.png')
        plt.close()

        print "day trade time: ", daytime
    print "total tradetime: ", tradetime
