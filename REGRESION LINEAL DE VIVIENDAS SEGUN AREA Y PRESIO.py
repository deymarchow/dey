import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x=[2104.0,1600.0,2400.0,1416.0,3000.0]
y=[400.0,330.0,369.0,232.0,540.0]
n=len(x)
x=np.array(x)
y=np.array(y)
#sumatorias
sumx=sum(x)
print ('es lavor de sumatoria de "x" es:' "%.2f"%sumx)
sumy=sum(y)
print ('es lavor de sumatoria de "y" es:' "%.2f"%sumy)
sumx2=sum(x*x)
print ('es lavor de sumx2 es:' "%.2f"%sumx2)
sumy2=sum(y*y)
print ('es lavor de sumy2 es:' "%.2f"%sumy2)
sumxy=sum(x*y)
print ('es lavor de sumxy es:'"%.2f"%sumxy)
print ('es lavor de sumx es:' "%.2f"%sumx)
#promedios
promx=sumx/n
promy=(sumy/n)
print ('el promedio de "X" es:' "%.2f"%promx)
print ('el promedio de "Y"es:' "%.2f"%promy)
# recta
m=(sumx*sumy-n*sumxy)/(sumx*sumx-n*sumx2)
print ('el valor de "M" es: ' "%.2f"%m)
b=(promy-(m*promx))
print ('el valor de "B" es: '"%.2f"%b)
plt.plot(x,y,"o",label="datos")
plt.plot(x,m*x+b, label="ajuste")
plt.xlabel("x")
plt.xlabel("y")
plt.title("regrecion lineal")
plt.grid()
plt.legend()
plt.show()