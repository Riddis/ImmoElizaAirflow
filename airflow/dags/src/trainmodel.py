from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import datetime

def prep_data(csv):
    """Select which columns to use as features, split the dataset, and scale it"""
    y = csv['price'].to_numpy() 

    cat_cols = ['property_type','kitchen','building_state','province', 'digit']
    numerical_cols = ['number_rooms', 'living_area', 'surface_land', 'number_facades', 'terrace', 'terrace_area', 'garden', 'garden_area']

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(csv[cat_cols])
    onehotdata = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(cat_cols))

    X = np.hstack([csv[numerical_cols], onehotdata])

    # Split the data into test and training sets and scale it
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, y, scaler, encoder

def train_XGBRegressor(X_train, y_train): 
    """Initializes the model"""
    return XGBRegressor(objective ='reg:squarederror', n_estimators = 50, seed = 123).fit(X_train, y_train)

def build_path():
    """Builds path to csv locations"""
    cwd = Path.cwd()
    date = datetime.date.today()
    csv_cleaned_path = f'dags/data/dataframe_cleaned_model_{date}.csv'
    src_path = (cwd / csv_cleaned_path).resolve()

    return src_path

def get_csv(src_path):
    """Parse the csv located at 'data/dataframe.csv'"""
    csv = pd.read_csv(src_path, index_col=0)

    return csv

def train():
    src_path = build_path()
    csv = get_csv(src_path)
    X_train, X_test, y_train, y_test, y, scaler, encoder = prep_data(csv)
    regressor = train_XGBRegressor(X_train, y_train)
    date = datetime.date.today()
    regressor.save_model(f'dags/models/xgbmodel_{date}.model')
    scaler_filename = f"dags/models/scaler_{date}.save"
    joblib.dump(scaler, scaler_filename) 
    encoder_filename = f"dags/models/encoder_{date}.save"
    joblib.dump(encoder, encoder_filename)