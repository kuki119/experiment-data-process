
import math

def cal(x1,y1,x2,y2):
    k=(y2-y1)/(x2-x1)
    A=math.atan(k)
    a=A*180/math.pi
    
    print A,a
    return a 





a = cal(1,1,2,2)


