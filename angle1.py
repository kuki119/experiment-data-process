
x1=float(input('please input the x1='))
y1=float(input('please input the y1='))
a=[x1,y1]
x2=float(input('please input the x2='))
y2=float(input('please input the y2='))
b=[x2,y2]
print a,b


k=(y2-y1)/(x2-x1)
import math 
A=math.atan(k)
a=A*180/math.pi
print 'a=',a