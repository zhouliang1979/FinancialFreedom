#!/bin/sh

rm -f nohup.out
#python demo.py --category rb1801 --start 20170823 --end 20170901 --barsize 1800 --ma  35 --bandwidth 20 --tickprice 1
nohup python test_case_reverse.py --category rb1801 --start 20170823 --end 20170901 --tickprice 1  &

