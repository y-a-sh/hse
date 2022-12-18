import pickle
import random
import pandas as pd
import re

model_data = None

def init_model():
    global model_data
    model_data = pickle.load(open('model.pickle', 'rb'))
    print('Model initialized')

def normalize_measure_units(df):    
    df['mileage'] = df['mileage'].str[0:-5].astype(float)
    df['engine'] = df['engine'].str[0:-3].astype(float)
    df['max_power'] = pd.to_numeric(df['max_power'].str[0:-4], errors='coerce')

    torq1 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]*(?:nm)?[ ]*(?:@|at|/)[ ]?(\d+[,.]?\d+)[-~](\d+[,.]?\d+)[ ]*(?:rpm)?', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq1[1] = (torq1[1] + torq1[2])/2
    torq1.drop(2, axis = 1, inplace = True)
    torq2 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]*(?:nm)?[ ]*(?:@|at|/)[ ]*(\d+[,.]?\d+)[ ]*(?:rpm)?', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq3 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]*(?:@|at|/)[ ]?(\d+[,.]?\d+)[,.]?-[,.]?(\d+[,.]?\d+)[ ]*\(kgm[ ]*@[ ]?rpm\)', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq4 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]*kgm (?:@|at|/) (\d+[,.]?\d+)[,.]?-[,.]?(\d+[,.]?\d+)[ ]*(?:rpm)?', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq3 = torq3.fillna(torq4)
    torq3[1] = (torq3[1] + torq3[2])/2
    torq5 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]?kgm (?:@|at|/) (\d+[,.]?\d+)[,.]?[ ]?rpm', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq6 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]*(?:@|at|/)[ ]*(\d+[,.]?\d+)[,.]??\(kgm[ ]*(?:@|at)[ ]?rpm\)', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq7 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]?kgm[ ]?(?:@|at|/)[ ]?(\d+[,.]?\d+)[,.]?', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq8 = df['torque'].str.extract('(\d+[,.]?\d+)[ ]*nm$', flags=re.IGNORECASE).replace(',', '', regex=True).astype(float)
    torq3 = torq3.fillna(torq5).fillna(torq6).fillna(torq7).fillna(torq8)

    torq3[0] = torq3[0] * 9.80665
    torq3.drop(2, axis = 1, inplace = True)
    torq1 = torq1.fillna(torq2).fillna(torq3)

    df.drop(columns = ['torque'])
    df['torque'] = torq1[0]
    df['max_torque_rpm'] = torq1[1]

def feature_eng(df):
    return pd.concat([df, df['year']**2, df['max_power']**2, df['engine']**2, df['max_power']/df['engine']], axis=1)

def predict_df(df):
    df = df.copy()
    normalize_measure_units(df)
    cat_features_mask = (df.dtypes == "object").values
    df_real = df[df.columns[~cat_features_mask]]
    df_real = pd.DataFrame(data=model_data['mis_replacer'].transform(df_real), columns=df_real.columns)
    df_cat = df[df.columns[cat_features_mask]].fillna("")
    df_cat.reset_index(drop=True, inplace=True)
    df_no_mis = pd.concat([df_real, df_cat], axis=1)
    df_no_mis['seats'] = df_no_mis['seats'].astype('int')
    df_no_mis['year'] = df_no_mis['year'].astype('int')
    df_no_mis['engine'] = df_no_mis['engine'].astype('int')
    cat_no_mis_features_mask = (df_no_mis.dtypes == "object").values
    X_real = df_no_mis[df_no_mis.columns[~cat_no_mis_features_mask]].drop('selling_price', axis=1)
    X_real_scaled = pd.DataFrame(data=model_data['scaler'].transform(X_real), columns = X_real.columns)
    X_with_cat = pd.concat([X_real_scaled, df_no_mis[['fuel', 'seller_type', 'transmission', 'owner']]], axis=1)
    X_feat = pd.concat([feature_eng(X_with_cat), df['name'].str.split(' ').str.get(0)], axis=1)
    X_feat = model_data['encoder'].transform(X_feat)
    return model_data['model'].predict(X_feat)

