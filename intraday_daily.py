# coding: utf-8

import pandas as pd
import argparse
import csv
import numpy as np
import sys
import math

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


def enterobservation(tickbuff, flag, args):
    # 开盘即入场
    if len(tickbuff) < 2:
        if flag == UP:
            return LONG
        if flag == DOWN:
            return SHORT

    if len(tickbuff) < 60*120:
        return 0

    if flag == UP and max(tickbuff[-60*120:]) < tickbuff[0]:
        return SHORT
    if flag == DOWN and min(tickbuff[-60*120:]) > tickbuff[0]:
        return LONG

    return 0


def earn(price1, price2, dire):
    res = price1-price2 if dire > 0 else price2-price1
    return res


def leaveobservation(dire, tickbuff, tradebuff, args):
    thresh = args.thresh
    bestprice = max(tradebuff) if dire == LONG else min(tradebuff)
    obvlen = 30
    if len(tickbuff)-len(tradebuff) < 3:
        if earn(tradebuff[-1], tradebuff[0], dire) > thresh*args.tickprice \
                and earn(tradebuff[-1], bestprice, dire) < -5*args.tickprice:
            return True
        if len(tickbuff) < 60*120:
            return False
        if dire == LONG and max(tickbuff[-60*120:]) < tickbuff[0]:
            return True
        if dire == SHORT and min(tickbuff[-60*120:]) > tickbuff[0]:
            return True

    # 1小时反向入场的出场策略, 等到收盘
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
        "--thresh",
        type=int,
        default=10)
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
    prevflag = None
    flagenter = flaglea = None
    tradetime = 0
    for data in datas:
        if prevflag is None:
            if data.price[-3] > data.price[5]:
                prevflag = UP
            else:
                prevflag = DOWN
            continue
        tradebuff = []      # 记录开仓后所有的tick数据
        tickbuff = []       # 记录所有的tick价位
        longmarks = []      # 记录做多时点
        shortmarks = []     # 记录做空时点
        closemarks = []     # 记录平仓时点
        daytime = 0         # 记录每天的交易次数

        # 输出策略日期
        days = data.index[-1]
        print "day: ", days
        # 记录一天数据长度， 最后半小时， 不再开仓
        daylen = len(data)
        for i in range(5, len(data)):
            item = data.iloc[i]
            tickbuff.append(item.price)
            nowprice = item.price

            # 记录自开仓以来的所有tick， 平仓后清空
            if dire_ is not None:
                tradebuff.append(item.price)

            if dire_ is None:
                flagenter = enterobservation(tickbuff, prevflag, args)
                if flagenter == LONG and i < len(data)-300:
                    price_ = item.ask_price1
                    bestprice_ = price_
                    dire_ = LONG
                    longmarks.append((i, price_))
                    tradebuff.append(item.price)
                    daytime += 1
                    continue
                elif flagenter == SHORT and i < len(data)-300:
                    price_ = item.bid_price1
                    bestprice_ = price_
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
                flaglea = leaveobservation(dire_, tickbuff, tradebuff, args)
                if flaglea or i >= len(data)-300:
                    if dire_ == LONG:
                        oneprofit = item.bid_price1-price_
                        profit += oneprofit
                        print "profit: ", oneprofit, profit
                        price_ = item.bid_price1
                        dire_ = None
                        closemarks.append((i, price_))
                        tradebuff = []
                        tradetime += 1
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
                        continue

        # 更新日bar的开高低收
        prevflag = UP if data.price[-1] > data.price[0] else DOWN
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
