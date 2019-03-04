import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading data
customers = pd.read_csv('Ecommerce Customers')

customers.head()
customers.describe()
customers.info()

#exploratory data analysis
sns.set()
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)
plt.show()
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)
plt.show()
sns.jointplot(x="Time on App", y="Length of Membership", data=customers, kind='hex')
plt.show()
sns.pairplot(customers)
plt.show()
sns.lmplot(x="Length of Membership",y="Yearly Amount Spent",data=customers)
plt.show()

#training and testing data
X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
coeff = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
print(coeff)

#predicting test data
pred = lm.predict(X_test)
plt.scatter(y_test,pred,data=customers)
plt.show()

#evaluating the model
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

sns.distplot((y_test-pred),bins=50)
plt.show()