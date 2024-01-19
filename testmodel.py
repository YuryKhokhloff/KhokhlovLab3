import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('/home/yury/Загрузки/salarylab2.csv', decimal=',')
X=df[['age','education']]
y=df['salary']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

lm=LinearRegression()
lm.fit(X_train,y_train)

pickle.dump(lm, open('modellab2.pickle','wb'))