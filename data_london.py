import numpy as np
import pandas as pd

np.random.seed(123)

###
#Load and process data
###

df = pd.read_table('lpmc.dat')

df.sort_values(['household_id', 'person_n', 'trip_n'], inplace=True)

df = df[df['purpose'] <= 3]
df = df[df['age'] >= 12]

df.reset_index(drop=True, inplace=True)
    

df['household_id'] += 1
df['person_n'] += 1
df['person_n'].max()

df['ind_id'] = df['household_id'] * 1000 + df['person_n']

#Sample observations
inds = df['ind_id'].unique()
n_ind = inds.shape[0]
n_use = 6250
inds_use = np.random.choice(inds, size=n_use, replace=False)
df = df[df['ind_id'].isin(inds_use)].reset_index(drop=True)

#Create new ind id
n_obs = df.shape[0]
n_alt = 4
new_ind_id = np.ones((n_obs,1), dtype='int64')
for i in np.arange(1,n_obs):
    if df['ind_id'].iloc[i] == df['ind_id'].iloc[i - 1]:
        new_ind_id[i] = new_ind_id[i - 1]
    else:
        new_ind_id[i] = new_ind_id[i - 1] + 1

df['ind_id'] = new_ind_id

inds = df['ind_id'].unique()
n_ind = inds.shape[0]
n_obs = df.shape[0]
df['obs_id'] = np.arange(1, n_obs+1)

n_train = round(0.8 * n_ind)
n_val = n_ind - n_train
inds_train = np.random.choice(inds, size=n_train, replace=False)
inds_test = np.setdiff1d(inds, inds_train)

df['train_set'] = np.where(df['ind_id'].isin(inds_train), 1, 0)
obs_test = [int(np.random.choice(df[df['ind_id'] == i]['obs_id'].values, 
                                 size=1, replace=False)) for i in inds_test]
df = df[np.logical_or(df['train_set'], df['obs_id'].isin(obs_test))].copy()

n_obs = df.shape[0]
df['obs_id'] = np.arange(1, n_obs+1)

for b in [True, False]:
    print(df['travel_mode'].value_counts(normalize=b))

###
#Wide to long
###

df_long = pd.DataFrame()

for j in range(1, n_alt+1):    
    
    df_tmp = pd.DataFrame()
    
    df_tmp['ind_id'] = df['ind_id']
    df_tmp['obs_id'] = np.arange(1, n_obs+1)
    df_tmp['alt_id'] = j
    
    df_tmp['chosen'] = 1 * (df['travel_mode'] == j)
    
    df_tmp['train_set'] = df['train_set']
    
    if j == 1:
        df_tmp['ovtt'] = df['dur_walking']
    elif j == 2:
        df_tmp['ovtt'] = df['dur_cycling']
    elif j == 3:
        df_tmp['ovtt'] = df['dur_pt_access'] #+ df['dur_pt_int']
        df_tmp['ivtt'] = df['dur_pt_rail'] + df['dur_pt_bus']
        df_tmp['transfers'] = df['pt_interchanges']
        df_tmp['cost'] = df['cost_transit']
    else:
        df_tmp['ivtt'] = df['dur_driving']
        df_tmp['cost'] = df['cost_driving_fuel'] #+ df['cost_driving_ccharge']
        #df_tmp['cost_driving_fuel'] = df['cost_driving_fuel']
        #df_tmp['cost_driving_ccharge'] = df['cost_driving_ccharge']
        df_tmp['traffic_var'] = df['driving_traffic_percent']
        
        df_tmp['driving_license'] = df['driving_license']
        df_tmp['car_ownership'] = df['car_ownership']
        
    df_tmp['age'] = df['age']
    df_tmp['female'] = df['female']
    df_tmp['winter'] = 1 * df['travel_month'].isin([10, 11, 12, 1, 2, 3])
        
    df_long = pd.concat((df_long, df_tmp), ignore_index=True)
 
    
df_long.fillna(0, inplace=True)

df_long.sort_values(['obs_id', 'alt_id'], inplace=True)
df_long.reset_index(drop=True, inplace=True)

df_long['tt'] = df_long['ovtt'] + df_long['ivtt']
"""
df_long['log_ovtt'] = np.log(1 + df_long['ovtt'] * 60)
df_long['log_ovtt'] -= df_long['log_ovtt'].mean()
df_long['log_ivtt'] = np.log(1 + df_long['ivtt'] * 60)
df_long['log_ivtt'] -= df_long['log_ivtt'].mean()
"""

df_long.to_csv('london.csv', index=False)

