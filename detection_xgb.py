import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as nw
import pandapower.estimation as est
import pickle
from keras.models import Sequential
from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn import linear_model
from sklearn.metrics import mean_squared_error
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
from keras.models import load_model

net = nw.case33bw()

pp.create_gen(net, 21, p_mw =0.2,q_mvar =0.05, type ="PV", in_service =True)
pp.create_gen(net, 31, p_mw =0.5,q_mvar =0.02, type ="PV", in_service =True)
pp.create_gen(net, 7, p_mw =0.2,q_mvar =0.05, type ="PV", in_service =True)
pp.create_gen(net, 15, p_mw =0.1,q_mvar =0.01, type ="PV", in_service =True)

pp.runpp(net)
ybus = net._ppc["internal"]["Ybus"].todense()
# print(ybus)
G = (ybus.real)
B = ybus.imag

attk_bus = [10,11,12,19,22,23,25,26,27,29,30]
attack_instants =[3,10,22,48,49,40]
# meas_with_optmz_attk = pd.read_csv('./meas_with_optmz_fdia.csv').drop('Unnamed: 0', axis ="columns")
meas_with_base_attk = pd.read_csv('./meas_data_with_base_attk.csv').drop('Unnamed: 0', axis ="columns")

# model_path = './ada_regressor_try.pkl'

# for SE through ML
col_p =[]
col_q =[]
for n in range(33):
    temp_p = 'p' +str(n)
    temp_q = 'q' + str(n)
    col_p.append(temp_p)
    col_q.append(temp_q)

col_pq = col_p + col_q

col_p_attk =[]
col_q_attk =[]
for n in attk_bus:
    temp_p = 'p' +str(n)
    temp_q = 'q' + str(n)
    col_p_attk.append(temp_p)
    col_q_attk.append(temp_q)
# print(col_p_attk)

col_opt_attk =col_p_attk +(col_q_attk)

meas_base_attk_fr_ml = meas_with_base_attk[col_pq]

# scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = joblib.load('scaler_xgb.pkl')


# ========================for base attack detection=============================================
# -----------------------------------------------------------WLS---------------------------------------------------------------------

def state_estimation(z_a,t):
    print(t)
    # data = pd.DataFrame(z_a.iloc[t,:]).T
    
    data = z_a
    # print(data.columns)
    column_a = []
    column_vm = []


    for n in range(len(net.bus.index)):
       column_a.append('a' + str(n))
       column_vm.append('v' + str(n))
       se_a = pd.DataFrame(columns=column_a)
       se_v = pd.DataFrame(columns=column_vm)

# print(se_a)
    # print(se_v)

    # for t in range(1):
    for n in range(len(net.bus.index)):
       
        column_v = 'v' + str(n)
        
        column_q = 'q' + str(n)
       
        column_p = 'p' + str(n)
        
        
        column_pl = 'pl' + str(n)
        column_ql = 'ql' + str(n)
        bus_no = n
        # measuremnt_v_pu = (data._get_value(t, column_v))
        measuremnt_v_pu = 1
        # print(f"v is {measuremnt_v_pu}")
        measuremnt_q_mvar = (data._get_value(t, column_q))
        # print(f"q = {measuremnt_q_mvar}")
        measuremnt_p_mw = (data._get_value(t, column_p))
       
       
        # measuremnt_pl_mw = data._get_value(t, column_pl)
        # measuremnt_ql_mw = data._get_value(t, column_ql)

        pp.create_measurement(
            net, 'v', 'bus', measuremnt_v_pu, .01, bus_no)
        pp.create_measurement(
            net, 'p', 'bus',  measuremnt_p_mw, 0.01, bus_no)
        pp.create_measurement(
            net, 'q', 'bus', measuremnt_q_mvar, 0.01, bus_no)
        

    for l in range(len(net.line.index)):
       
        column_pl = 'pl' + str(l)
        column_ql = 'ql' + str(l)

        measuremnt_pl_mw = (data._get_value(t, column_pl))
        # print(f"pl is {measuremnt_pl_mw}")
        
        measuremnt_ql_mvar = (data._get_value(t, column_ql))

        pp.create_measurement(
             net, 'q', 'line', measuremnt_ql_mvar, 0.02, l, side='from')
        pp.create_measurement(
                net, 'p', 'line', measuremnt_pl_mw, 0.01, l, side='from')

    # success = est.estimate(net, init="flat")
    est.estimate(net, init ='flat')
    # print('success')
    print(pp.diagnostic(net, report_style='detailed') )
    
    state_values = [net.res_bus_est['vm_pu'], net.res_bus_est['va_degree']]
    se_v.loc[t] = np.array(net.res_bus_est['vm_pu'])
    se_a.loc[t] = np.array(net.res_bus_est['va_degree'])
    # se_va_degree = net.res_bus_est['va_degree']

    x_est = pd.concat([se_a,se_v],axis = 1)
    
    return se_v, se_a   



model_path = './xgb_model.pkl'

neural_model =joblib.load(model_path)


x= range(33)
mse =[]
for t in range(50):
    # ------------------------------------for wls----------------------------------------------------------------------
    meas_to_se = pd.DataFrame(meas_with_base_attk.iloc[t, :]).T.reset_index()
    # print(meas_to_se)
    v_est, theta_est = state_estimation(meas_to_se, 0)
    v_est_arr = np.array(v_est).flatten()
    
    # ------------for ml---------------------------------
    x_pq = pd.DataFrame(meas_base_attk_fr_ml.iloc[t,:]).T
    
    x_scaled = scaler.transform(x_pq)
    print("data_to_model", x_scaled)
    pred = neural_model.predict(x_scaled)

    v_pred = pred.flatten()
    mse_dt = mean_squared_error(v_est_arr, v_pred)
    print(mse_dt)
    mse.append(mse_dt)
    
    # ----------------------compare wls and ml-------------------------------------
   
file_path = "mse_base_attk_std_scaler_xgb.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(mse, file)

print(f"list saved to {file_path}")

