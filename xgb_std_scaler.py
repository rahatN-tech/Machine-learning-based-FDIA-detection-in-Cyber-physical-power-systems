from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import linear_model
from sklearn.metrics import mean_squared_error
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import time
import joblib


p_inj = pd.read_csv('p_inj_1_yr.csv').drop('Unnamed: 0', axis ='columns')
q_inj = pd.read_csv('q_inj_1_yr.csv').drop('Unnamed: 0', axis ='columns')
v = pd.read_csv('v_est_1_yr.csv').drop('Unnamed: 0', axis ='columns')

pq_inj = pd.concat([p_inj, q_inj], axis =1)
# print(len(pq_inj))
pq_inj_model =pq_inj.iloc[:35000]
pq_inj_verify = pq_inj.iloc[35000:]

v_model = v.iloc[:35000]
v_verify = v.iloc[35000:]

X= pq_inj_model
# X = p_inj
y= v_model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=20)

# print(y_train.shape)
y_train_reshaped = y_train.values.reshape(-1,)  # Reshape to 1D array
y_test_reshaped = y_test.values.reshape(-1,)
# print(y_train_reshaped.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# scaler =StandardScaler()
# scaler.fit_transform(X_train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
x_verify = scaler.transform(pq_inj_verify)
# print(x_verify.shape)
joblib.dump(scaler, './scaler_xgb.pkl')

# gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
start_time = time.time()
xgb_regressor = XGBRegressor( n_estimators=100, learning_rate=0.01, random_state=0)
xgb_multi_regressor = MultiOutputRegressor(xgb_regressor)

# Fit the MultiOutputRegressor model
xgb_multi_regressor.fit(X_train_scaled, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"training time",training_time)

# import joblib


model_path = './xgb_model.pkl'

# Save the MultiOutputRegressor
# joblib.dump(xgb_multi_regressor, model_path)

print("Model saved successfully.")

y_pred = xgb_multi_regressor.predict(X_test_scaled)

# y_pred will have shape (n_samples_test, n_outputs)
print(y_pred.shape)

x= range(33)
v_wls = np.array(v_verify.iloc[0,:]).flatten()

mse_dt = mean_squared_error(v_wls, y_pred[0])
print(mse_dt)

plt.scatter(x, y_pred[0], label="predicted")
plt.scatter(x, v_wls, label= 'wls')
plt.legend()
plt.show()




    

# model.save('./model_minmax.h5')

# with open('./history_minmax.pkl', 'wb') as file:
#     pickle.dump(history.history, file)

# Create AdaBoostRegressor with GradientBoostingRegressor as base estimator
# gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
# ada_regressor = AdaBoostRegressor(base_estimator=gb_regressor, n_estimators=100, learning_rate=0.1, random_state=0)

# # Fit the AdaBoostRegressor model
# ada_regressor.fit(X_train_scaled, y_train)
