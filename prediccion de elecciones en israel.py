import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.svm import SVC

D_israel=pd.read_csv('results_by_booth_2013 - english - v2.csv')
# trabaje con drop pero me equivoque puesto que solo elimina filas duplicadas
# y no asi las duplicadas de una columna
"""
df=pd.DataFrame(D_israel)
df.drop_duplicates(['Ale Yarok','Brit Olam'],keep=False,inplace= True)
print (df)
"""
"""
df.drop_duplicates(['Ale Yarok','Am Shalem','Balad','Brit Olam','Daam Workers Party','Eretz Hadasha','Green Party','Hadash','Haim Bekavod','Hatnua','Hope for Change','HaYisraelim','Kadima','Kalkala','Koah LeHashpia','The Jewish Home','Labour Party','Leader','Likud Beitenu','Meretz','Moreshet Avot','Na Nach','Or','Otzma LeYisrael','Pirate Party','Raam-Taal','Senior Citizens Party','Shas','Social Justice','United Torah Judaism','Were Brothers','Yesh Atid'],keep=False, inplace=True)
print (df)
"""
#una manera de acortar los datos y teniendo mucha perdida de datos con (set)

# segui intentando con revome de varias maneras y no noto un cambio 
#mas al contrario el error continua
names=list (D_israel)  #en esta linea obtengo los nombres o los titulos de cada columna y las almaceno en una variable

column=(D_israel.Brit_Olam)    #en esta linea extraigo una columna de la base de datos de israel
                                #la guardo en una variable en tipo ista.  

q=np.array(column)      # lo converti en una array para poder remover los ceros pero me sale error 
                        
q.remove(0)     # AQUI EL ERROR
"""
q.remove(0)   #para esta linea deceo remover los " ceros" (0) de la columna seleccionada pero me lanza un error
"""
"""
X = D_israel.drop(['settlement_code','booth_number','Registered_voters'], axis = 1)
Y = D_israel['Registered_voters']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
print (Y_pred)
print (len(Y))
print (len (Y_pred))

plt.scatter(Y_test, Y_pred)
plt.xlabel("votos_registrados")
plt.ylabel("prediccion_de_votos")
plt.title("elecciones de israel")
plt.show()
"""
