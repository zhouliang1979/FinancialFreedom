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


def enterobservation(barlist, tickbuff, args):
    prevbuff = np.array(tickbuff[-300:])
    #maxvalue = barlist[-1][1] + 5*args.tickprice 
    #minvalue = barlist[-1][2] - 5*args.tickprice 
    maxvalue = np.mean(prevbuff)+3*np.std(prevbuff)
    minvalue = np.mean(prevbuff)-3*np.std(prevbuff)
    if tickbuff[-1] > maxvalue:
        if sum(prevbuff < maxvalue-1*args.tickprice) > 0 \
                and tickbuff[-1] <= maxvalue+2*args.tickprice:
            return SHORT

    if tickbuff[-1] < minvalue:
        if sum(prevbuff > minvalue+1*args.tickprice) > 0 \
                and tickbuff[-1] >= minvalue-2*args.tickprice:
            return LONG
    return None


def earn(price1, price2, dire):
    res = price1-price2 if dire > 0 else price2-price1
    return res


def leaveobservation(dire, barlist, tickbuff, tradebuff, i, lastenter,  args):
    tradeprice = tradebuff[0]
    r=600.0/(i-lastenter) if i-lastenter>600 else 1
    # 获利离场
    if earn(tickbuff[-1], tradeprice, dire) >= 5*r*args.tickprice:
        return True
    # 止损
    if earn(tickbuff[-1], tradeprice, dire) <= -6*r*args.tickprice:
        return True
        
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
        "--barwidth",
        type=int,
        default=600)
    parser.add_argument(
        "--mawidth",
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
    price_ = dire_ = None
    barlist = []        # 记录分钟级别bar的开高低收
    barclose = []       # 记录分钟级别bar的收盘价
    tradebuff = []      # 记录开仓后所有的tick数据
    tickbuff = []       # 记录所有的tick价位
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
            maxtar = barlist[-1][1] + 5*args.tickprice 
            mintar = barlist[-1][2] - 5*args.tickprice 
            beginday = False
            continue
        longmarks = []      # 记录做多时点
        shortmarks = []     # 记录做空时点
        closemarks = []     # 记录平仓时点
        daytime = 0         # 记录每天的交易次数

        # 存储必要的数据即可
        if len(tickbuff) > 10000:
            del tickbuff[:5000]

        # 输出策略日期
        # days = data.index[-1].strftime("%Y%m%d")
        days = data.index[-1]
        print "day: ", days
        print barlist[-1], maxtar, mintar
        daylen = len(data)
        lastenter = None
        for i in range(len(data)):
            item = data.iloc[i]
            tickbuff.append(item.price)
            nowprice = item.price

            # 记录自开仓以来的所有tick， 平仓后清空
            if dire_ is not None:
                tradebuff.append(item.price)

            # 30分钟内开始冷启动
            if len(tickbuff) < 600:
                continue
        
            # 已有持仓后，不在根据反方向信号离场， 而是根据leave逻辑
            # 时刻准备离场
            if dire_ is not None:
                # print "flagenter: ", flagenter
                flaglea = leaveobservation(fakedire_, barlist, tickbuff, tradebuff, i, lastenter, args)
                if flaglea or i >= len(data)-600:
                    if dire_ == LONG:
                        oneprofit = item.bid_price1-price_
                        profit += oneprofit
                        if flaglea:
                            print "GOOD LONG bid price profit: ", item.bid_price1, price_, oneprofit, profit
                        else:
                            print "BAD LONG bid price profit: ", item.bid_price1, price_, oneprofit, profit
                        price_ = item.bid_price1
                        dire_ = None
                        closemarks.append((i, price_))
                        tradebuff = []
                        tradetime += 1
                        allowopen = False
                        prevopen = i
                        prevdire = LONG
                        continue
                    else:
                        oneprofit = price_-item.ask_price1
                        profit += oneprofit
                        if flaglea:
                            print "GOOD SHORT price ask profit: ",price_, item.ask_price1, oneprofit, profit
                        else:
                            print "BAD SHORT price ask profit: ",price_, item.ask_price1, oneprofit, profit
                        price_ = item.ask_price1
                        dire_ = None
                        closemarks.append((i, price_))
                        tradebuff = []
                        tradetime += 1
                        allowopen = False
                        prevopen = i
                        prevdire = SHORT
                        continue
            
            # 最后半小时， 不再开仓
            if i > len(data)-600:
                continue

            if dire_ is None:
                flagenter = enterobservation(barlist, tickbuff, args)

                if flagenter == LONG: 
                    price_ = item.price
                    dire_ = LONG
                    fakedire_ = SHORT
                    longmarks.append((i, price_))
                    tradebuff.append(price_)
                    lastenter = i
                    daytime += 1
                    continue
                elif flagenter == SHORT:
                    price_ = item.price
                    dire_ = SHORT
                    fakedire_ = LONG
                    shortmarks.append((i, price_))
                    tradebuff.append(price_)
                    lastenter = i
                    daytime += 1
                    continue
                else:
                    continue


        # 更新日bar的开高低收
        barlist.append((data.price[0], max(data.price),
                        min(data.price), data.price[len(data)-1]))
        maxtar = barlist[-1][1] + 5*args.tickprice 
        mintar = barlist[-1][2] - 5*args.tickprice 
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
