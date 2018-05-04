#!/usr/bin/env python
#coding=utf-8
#author='J.Dai'

import time

def test():
    now_time = time.time()
    print("Now time:%d"%now_time)
    a = 10
    a += 1
    b = 'test string'
    print("a is", a)
    print("b is", b)

if __name__ == '__main__':
    test()
    print("Hello world!")
    print("Hello Git!")
    print("Bye-Bye")
    print("Time is incorrect!")
