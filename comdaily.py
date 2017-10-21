# coding: utf-8

import argparse
import numpy as np
import sys
import math
import pandas as pd
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
            if not line['bid_price1']:
                continue
            line['price'] = float(line['price'])
            line['bid_price1'] = float(line['bid_price1'])
            line['ask_price1'] = float(line['ask_price1'])
            line['bid_volume1'] = float(line['bid_volume1'])
            line['ask_volume1'] = float(line['ask_volume1'])
            fitcontent.append(line)
        if TradingDay > interval[1]:
            break
    df = pd.DataFrame(fitcontent)
    df = df.set_index('UpdateTime')
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


def enterobservation(barlist, tickbuff, nindex, pindex, args):
    # if len(barlist) < 3:
    #     return False
    prevbuff = np.array(tickbuff[-1200:])
    maxtar = max([barlist[-1][0], barlist[-1][-1]])
    mintar = min([barlist[-1][0], barlist[-1][-1]])
    # maxvalue = maxtar+(maxtar-mintar)/3.0
    # minvalue = mintar-(maxtar-mintar)/3.0
    intervals = [math.fabs(e[0]-e[-1]) for e in barlist[-3:]]
    interval = sum(intervals) / 6.0
    # maxvalue = maxtar + interval
    # minvalue = mintar - interval
    maxvalue = maxtar + 15
    minvalue = mintar - 15

    # 情况1，价位高于昨天的最高价(包括开盘价), 开多
    if tickbuff[-1] > maxvalue:
        if nindex <= 1801:
            return OLONG
        if sum(prevbuff < maxvalue-1*args.tickprice) > 0 \
                and tickbuff[-1] <= maxvalue+2*args.tickprice:
                # and sum(prevbuff > maxvalue+1*args.tickprice) == 0:
            return LONG

    # 情况2，价位高于昨天的最高价(包括开盘价), 开多
    if tickbuff[-1] < minvalue:
        if nindex <= 1801:
            return OSHORT
        if sum(prevbuff > minvalue+1*args.tickprice) > 0 \
                and tickbuff[-1] >= minvalue-2*args.tickprice:
                # and sum(prevbuff < minvalue-1*args.tickprice) == 0:
            return SHORT
    return 0


def earn(price1, price2, dire):
    res = price1-price2 if dire > 0 else price2-price1
    return res


def leaveobservation(dire, barlist, tickbuff, tradebuff, args, reverse=False):
    tradeprice = tradebuff[0]
    bestprice = max(tradebuff) if dire > 0 else min(tradebuff)
    # if earn(bestprice, tradeprice, dire) > 3*args.tickprice:
    if earn(tickbuff[-1], tradeprice, dire) <= -10*args.tickprice:
        return True
    return 0

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
        "--barwidth",
        type=int,
        default=600)
    parser.add_argument(
        "--ma1width",
        type=int,
        default=5)
    parser.add_argument(
        "--ma2width",
        type=int,
        default=13)
    parser.add_argument(
        "--ma3width",
        type=int,
        default=21)
    parser.add_argument(
        "--commision",
        type=float)
    args = parser.parse_args()
    if args.end is None:
        args.end = args.start

    datas = get_day_tick(args.category, (args.start, args.end), merge=False)

    global profit
    days = 0
    profit = 0
    bestprice_ = None
    price_ = dire_ = None
    barlist = []        # 记录分钟级别bar的开高低收
    barclose = []       # 记录分钟级别bar的收盘价
    tradebuff = []      # 记录开仓后所有的tick数据
    tickbuff = []       # 记录所有的tick价位
    ma1list = []        # 记录短期均线变化
    ma2list = []        # 记录长期均线变化
    ma3list = []        # 记录长期均线变化
    tradetime = 0
    interbuff = []      # 记录interest中值滤波后绝对变化量
    accinter = []       # 记录中值滤波后interest三分钟的累积量
    interblock = 360    # 记录三分钟的成交量， 可以调节
    medmodel = 5
    allowopen = True
    prevopen = -100
    prevdire = None
    beginday = True
    flagenter = flaglea = None
    for data in datas:
        # 记录第一天的开高低收后继续
        if beginday:
            barlist.append((data.price[0], max(data.price),
                            min(data.price), data.price[len(data)-1]))
            beginday = False
            continue
        longmarks = []      # 记录做多时点
        shortmarks = []     # 记录做空时点
        closemarks = []     # 记录平仓时点
        daytime = 0         # 记录每天的交易次数

        # 存储必要的数据即可
        if len(tickbuff) > 100000:
            del tickbuff[:50000]

        # 输出策略日期
        # days = data.index[-1].strftime("%Y%m%d")
        days = data.index[-1]
        print "day: ", days
        # 记录一天数据长度， 最后半小时， 不再开仓
        daylen = len(data)
        lastenter = None
        reverse = False
        for i in range(len(data)):
            item = data.iloc[i]
            tickbuff.append(item.price)
            nowprice = item.price

            # 记录自开仓以来的所有tick， 平仓后清空
            if dire_ is not None:
                tradebuff.append(item.price)

            if i > len(data)-600:
                continue
            if i < 1800:
                continue

            if dire_ is None:
                flagenter = enterobservation(barlist, tickbuff, i, lastenter, args)

                if flagenter == LONG or flagenter == OLONG:
                    price_ = item.price
                    bestprice_ = item.price
                    dire_ = LONG
                    longmarks.append((i, price_))
                    tradebuff.append(item.price)
                    daytime += 1
                    continue
                elif flagenter == SHORT or flagenter == OSHORT:
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
                # print "flagenter: ", flagenter
                flaglea = leaveobservation(dire_, barlist,
                                           tickbuff, tradebuff, args, reverse)
                if flaglea or i >= len(data)-600:
                    if dire_ == LONG:
                        oneprofit = item.bid_price1-price_
                        profit += oneprofit
                        print "profit: ", oneprofit, profit
                        price_ = item.bid_price1
                        dire_ = None
                        closemarks.append((i, price_))
                        tradebuff = []
                        tradetime += 1
                        allowopen = False
                        prevopen = i
                        prevdire = LONG
                        reverse = True if oneprofit < -0.01 else False
                        # if not reverse:
                        #     break
                        continue
                    else:
                        oneprofit = price_-item.ask_price1
                        profit += oneprofit
                        print "profit: ", oneprofit, profit
                        price_ = item.ask_price1
                        dire_ = None
                        closemarks.append((i, price_))
                        tradebuff = []
                        tradetime += 1
                        allowopen = False
                        prevopen = i
                        prevdire = SHORT
                        reverse = True if oneprofit < -0.01 else False
                        # if not reverse:
                        #     break
                        continue

        # 更新日bar的开高低收
        barlist.append((data.price[0], max(data.price),
                        min(data.price), data.price[len(data)-1]))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(200, 20))
        plt.plot(data.price.as_matrix())
        plt.plot([e[0] for e in longmarks], [p[1] for p in longmarks], 'r^')
        plt.plot([e[0] for e in shortmarks], [p[1] for p in shortmarks], 'yv')
        plt.plot([e[0] for e in closemarks], [p[1] for p in closemarks], 'ko')
        plt.savefig('mark_'+str(days)+'.png')
        plt.close()

        print "day trade time: ", daytime
    print "total tradetime: ", tradetime
