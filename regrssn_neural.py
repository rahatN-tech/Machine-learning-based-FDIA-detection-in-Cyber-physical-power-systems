import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
import time


p_inj = pd.read_csv('p_inj_1_yr.csv').drop('Unnamed: 0', axis ='columns')
q_inj = pd.read_csv('q_inj_1_yr.csv').drop('Unnamed: 0', axis ='columns')
v = pd.read_csv('v_est_1_yr.csv').drop('Unnamed: 0', axis ='columns')

pq_inj_temp = pd.concat([p_inj, q_inj], axis =1)
pq_inj = pq_inj_temp.drop(['p0','q0'], axis=1)



pq_inj_model =pq_inj.iloc[:35000]
pq_inj_verify = pq_inj.iloc[35000:]

v_model = v.iloc[:35000]
v_verify = v.iloc[35000:]

X= pq_inj_model
# X = p_inj
y= v_model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=20)


scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print("scaled\n:", (X_train_scaled[0,:]))

# defining the model
joblib.dump(scaler, 'scaler_neural.pkl')
model = Sequential()
model.add(Dense(128, input_dim =64, activation='relu'))
model.add(Dense(64, activation='relu'))
# output layer
model.add(Dense(33, activation='linear'))

model.compile(loss='mean_squared_error', optimizer ='adam', metrics =['mae'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True)

start_time = time.time()
history = model.fit(X_train_scaled, y_train, validation_split =0.2, epochs =1000, callbacks=[early_stopping])
end_time = time.time()

training_time = end_time-start_time

print(f"time taken {training_time}")




model.save('./model_std_scaler.h5')

with open('./history_regression_neural.pkl', 'wb') as file:
    pickle.dump(history.history, file)