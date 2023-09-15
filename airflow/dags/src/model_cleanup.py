import pandas as pd
import numpy as np
from pathlib import Path
import re

def build_path():
    """Builds path to csv locations"""
    cwd = Path.cwd()
    csv_path = 'dags/data/dataframe.csv'
    csv_cleaned_path = 'dags/data/dataframe_cleaned_model.csv'
    src_path = (cwd / csv_path).resolve()
    out_path = (cwd / csv_cleaned_path).resolve()

    return src_path, out_path

def get_csv(src_path):
    """Parse the csv located at 'data/dataframe.csv'"""
    csv = pd.read_csv(src_path, index_col=0)

    return csv

def convert(n):
    """Divides zipcode by 100 to generalise the data a little and prevent overfitting"""
    if n == 'other':
        return 'other'
    else:
        return str(int(int(n)/100))

def clean_csv(csv):
    """Removes duplicates and drops rows with emtpy cells"""
    # Removing unneeded data
    # dropping empty rows
    csv = csv.dropna(how='all')
    # dropping duplicates (if any)
    csv = csv.drop_duplicates()
    # Dropping house and appartment groups since they have no data
    csv = csv.drop(csv[(csv['property_type'] == 'HOUSE_GROUP') | (csv['property_type'] == 'APARTMENT_GROUP')].index)
    # Drop rows without a price property
    csv = csv.drop(csv[pd.isna(csv['price']) == True].index)
    # Drop rows with 0 rooms
    csv = csv.drop(csv[csv['number_rooms'] == 0].index)
    # Drop rows without a living area property
    csv = csv.drop(csv[pd.isna(csv['living_area']) == True].index)
    # Assuming that a NaN value or 0 means no kitchen installed, replacing the strings with integers
    # 0 = NOT_INSTALLED, 0.5 = SEMI_EQUIPPED, 1 = INSTALLED, 2 = HYPER_EQUIPPED
    csv['kitchen'] = csv['kitchen'].fillna('NOT_INSTALLED')
    csv['kitchen'] = csv['kitchen'].replace('0', 'NOT_INSTALLED')
    csv['kitchen'] = csv['kitchen'].replace(0, 'NOT_INSTALLED')
    """csv['kitchen'] = csv['kitchen'].replace('USA_UNINSTALLED', 0)
    csv['kitchen'] = csv['kitchen'].replace('SEMI_EQUIPPED', 0.5)
    csv['kitchen'] = csv['kitchen'].replace('USA_SEMI_EQUIPPED', 0.5)
    csv['kitchen'] = csv['kitchen'].replace('INSTALLED', 1)
    csv['kitchen'] = csv['kitchen'].replace('USA_INSTALLED', 1)
    csv['kitchen'] = csv['kitchen'].replace('HYPER_EQUIPPED', 2)
    csv['kitchen'] = csv['kitchen'].replace('USA_HYPER_EQUIPPED', 2)"""
    # Filling empty values and changing true/false to 1/0
    csv['furnished'] = csv['furnished'].fillna(0)
    csv['furnished'] = csv['furnished'].replace(False, 0)
    csv['furnished'] = csv['furnished'].replace(True, 1)
    # Assuming that a NaN value,0 or -1 means no fireplace installed
    csv['fireplace'] = csv['fireplace'].fillna(0)
    csv['fireplace'] = csv['fireplace'].replace(-1, 0)
    # Filling empty values and changing true/false to 1/0
    csv['terrace'] = csv['terrace'].fillna(0)
    csv['terrace'] = csv['terrace'].replace(False, 0)
    csv['terrace'] = csv['terrace'].replace(True, 1)
    # Assuming the surface area = living area in case of apartments
    to_replace = csv[((csv['surface_land'] == 'UNKNOWN')|(pd.isna(csv['surface_land']) == True)) & (csv['property_type'] == 'APARTMENT')]
    to_replace = to_replace.reset_index()
    # Looping through rows to replace the values
    for index, row in to_replace.iterrows():
        csv.loc[row['index'], 'surface_land'] = row['living_area']
    # Dropping rows with no surface area 
    csv = csv.drop(csv[(csv['surface_land'] == 'UNKNOWN') | (pd.isna(csv['surface_land']) == True) | (csv['surface_land'] == 0)].index)
    # Dropping rows with no facade info
    csv = csv.drop(csv[(csv['number_facades'] == 'UNKNOWN') | (pd.isna(csv['number_facades']) == True)].index)
    # Filling empty values and changing true/false to 1/0
    csv['swimming_pool'] = csv['swimming_pool'].fillna(0)
    csv['swimming_pool'] = csv['swimming_pool'].replace(False, 0)
    csv['swimming_pool'] = csv['swimming_pool'].replace(True, 1)
    csv = csv.drop(csv[(csv['building_state'] == 'UNKNOWN') | (pd.isna(csv['building_state']) == True)].index)
    # If terrace = 1 but no terrace_area present, drop the row
    csv = csv.drop(csv[(csv['terrace'] == 1) & (pd.isna(csv['terrace_area']) == True)].index)
    # Filling empty values and changing true/false to 1/0
    csv['terrace_area'] = csv['terrace_area'].fillna(0)
    # If garden = 1 but no garden_area present, drop the row
    csv = csv.drop(csv[(csv['garden'] == 1) & (pd.isna(csv['garden_area']) == True)].index)
    # No garden, filling empty values
    csv['garden'] = csv['garden'].fillna(0)
    csv['garden'] = csv['garden'].replace(False, 0)
    csv['garden'] = csv['garden'].replace(True, 1)
    csv['garden_area'] = csv['garden_area'].fillna(0)
    # Change strings to floats in certain columns
    csv = csv.drop(csv[(csv['surface_land'] == 0)].index)
    csv['surface_land']=csv['surface_land'].astype("float")
    csv['number_facades']=csv['number_facades'].astype("float")
    csv = csv.drop(csv[csv['zip_code'] == 'UNKNOWN'].index)
    csv = csv.drop(csv[pd.isna(csv['region']) == True].index)
    csv = csv.drop(csv[pd.isna(csv['province']) == True].index)
    csv['zip_code']=csv['zip_code'].astype("str")
    csv['ppm'] = csv['price']/csv['surface_land']
    # Removing zipcodes that are not 4 numbers
    patternDel = "\b[0-9]\{4\}\b"
    filter = csv['zip_code'].str.contains(patternDel)
    csv = csv[~filter]
    # If we have less than 3 occurences, zipcode will be changed to 'other' so we don't overfit
    filter = csv['zip_code'].value_counts()
    csv['zip_code'] = np.where(csv['zip_code'].isin(filter.index[filter >= 4]), csv['zip_code'], 'other')
    # Removing outliers
    cols = ['price', 'number_rooms', 'living_area',
        'furnished', 'fireplace', 'terrace', 'terrace_area', 'garden',
        'garden_area', 'surface_land', 'number_facades', 'swimming_pool'] # one or more
    Q1 = csv[cols].quantile(0.25)
    Q3 = csv[cols].quantile(0.75)
    IQR = Q3 - Q1
    csv = csv[~((csv[cols] < (Q1 - 1.5 * IQR)) |(csv[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    csv["digit"]=csv["zip_code"].agg(convert)

    return csv

def save_csv(csv, out_path):
    """Saves the cleaned up CSV to 'output/dataframe_cleaned.csv'"""
    csv.to_csv(out_path)

def clean():
    # Get path variables
    src_path, out_path = build_path()
    # Parse csv
    csv = get_csv(src_path)
    # Clean up the csv
    csv = clean_csv(csv)
    # Save the csv
    save_csv(csv, out_path)

clean()