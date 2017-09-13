# coding:utf-8

import math
import numpy as np
import time

x = [i*0.001 for i in xrange(1000000)]
start = time.clock()
for i,t in enumerate(x):#enumerate用于遍历x的元素及其下标，即i遍历下标，t遍历x中的元素
    x[i] = math.sin(t)
print 'math.sin:',time.clock()-start

x = [i*0.001 for i in xrange(1000000)]
x = np.array(x)
start = time.clock()
np.sin(x,x)
print 'numpy.sin:',time.clock()-start
