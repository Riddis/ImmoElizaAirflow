import streamlit as st
from pathlib import Path
from xgboost import XGBRegressor
import pandas as pd
import joblib
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json

def convert(n):
    """Divides zipcode by 100 to generalise the data a little and prevent overfitting"""
    if n == 'other':
        return 'other'
    else:
        return str(int(int(n)/100))

def preprocess_new_data(json_data, encoder, scaler):
    
    json_schema = {
        "type" : "object",
        "properties" : {
            "data" : {
                "type" : "object",
                "properties" : {
                    "area": {type: "integer"},
                    "property-type" : {
                        type: "string", 
                        "enum": ['APARTMENT', 
                                'HOUSE']
                    },
                    "rooms-number": {type: "integer"},
                    "zip-code": {type: "integer"},
                    "full-address": {type: "string"},
                    "land-area": {type: "integer"},
                    "garden": {type: "boolean"},
                    "garden-area": {type: "integer"},
                    "equipped-kitchen": {
                        type: "string",
                        "enum": ['NOT_INSTALLED', 
                                 'USA_UNINSTALLED', 
                                 'SEMI_EQUIPPED', 
                                 'USA_SEMI_EQUIPPED', 
                                 'INSTALLED', 
                                 'USA_INSTALLED',
                                 'HYPER_EQUIPPED', 
                                 'USA_HYPER_EQUIPPED'],
                    },
                    "province": {
                        type: "string",
                        "enum": ['Antwerp', 
                                 'Brussels', 
                                 'East Flanders', 
                                 'West Flanders', 
                                 'Flemish Brabant', 
                                 'Hainaut',
                                 'Liège', 
                                 'Limburg', 
                                 'Luxembourg', 
                                 'Namur', 
                                 'Waloon Brabant']
                     },
                    "swimming-pool": {type: "boolean"},
                    "furnished": {type: "boolean"},
                    "open-fire": {type: "boolean"},
                    "terrace": {type: "boolean"},
                    "terrace-area": {type: "integer"},
                    "facades-number": {type: "integer"},
                    "building-state": {
                        type: "string",
                        "enum": ['NEW', 
                                'GOOD', 
                                'RENOVATE', 
                                'JUST RENOVATED', 
                                'TO REBUILD']
                    },
                }, 
                "required": ["area", 
                             "property-type", 
                             "rooms-number", 
                             "zip-code"],
                "additionalProperties": False
            }
        },
        "required": ["data"],
        "additionalProperties": False
    }

    try:
        validate(instance=json_data, schema=json_schema)
    except ValidationError as error: 
        status = 400
        return error, status
        
    data = {}
    for key, value in json_data['data'].items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                data[sub_key] = sub_value
        else:
            data[key] = value

    df = pd.DataFrame(data, index=[0])
 

    # rename colums
    df = df.reset_index().rename(columns ={"area": "living_area", 
                                            "property-type": "property_type", 
                                            'rooms-number':'number_rooms', 
                                            'zip-code':'digit', 
                                            'land-area': 'surface_land', 
                                            'garden-area':'garden_area', 
                                            'equipped-kitchen':'kitchen', 
                                            'terrace-area':'terrace_area', 
                                            'facades-number':'number_facades', 
                                            'building-state':'building_state'})

    df["digit"]=df["digit"].agg(convert)
    df = df.drop(columns='index')

    cat_cols = ['property_type',
                'kitchen',
                'building_state',
                'province', 
                'digit']
    
    numerical_cols = ['number_rooms', 
                      'living_area', 
                      'surface_land', 
                      'number_facades', 
                      'terrace', 
                      'terrace_area', 
                      'garden', 
                      'garden_area']

    encoded_data = encoder.transform(df[cat_cols])
    onehotdata = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(cat_cols))

    X = np.hstack([df[numerical_cols], onehotdata])

    X = scaler.transform(X)

    status = 200
    return X, status

def build_path():
    """Builds path to csv locations"""
    cwd = Path.cwd()
    date = datetime.date.today()
    csv_cleaned_path = f'dags/data/dataframe_cleaned_visual_{date}.csv'
    return (cwd / csv_cleaned_path).resolve()

def get_csv(src_path):
    """Parse the csv located at 'data/dataframe.csv'"""
    return pd.read_csv(src_path, index_col=0)

def make_plots(df):
    plt.rcParams.update({'font.size': 20})

    fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.plot(df1.name, df1.users_count)
    ax1.set_title('Highest amount of users')
    ax1.set_ylabel('Users Count')
    ax1.set_xticklabels(df1.name, rotation=45, ha="right")

    ax2 = axes[0, 1]
    ax2.plot(df2.name, df2.users_count)
    ax2.set_title('Least amount of users')
    ax2.set_ylabel('Users Count')
    ax2.set_xticklabels(df2.name, rotation=45, ha="right")

    ax3 = axes[1, 0]
    ax3.plot(df3.name, df3.total_sold)
    ax3.set_title('Most sales')
    ax3.set_ylabel('Total Sales Volume')
    ax3.set_xticklabels(df3.name, rotation=45, ha="right")


    ax4 = axes[1, 1]
    ax4.plot(df4.name, df4.total_sold)
    ax4.set_title('Lowest sales')
    ax4.set_ylabel('Total Sales Volume')
    ax4.set_xticklabels(df4.name, rotation=45, ha="right")


    fig1.suptitle("Countries to focus our marketing budget on", fontsize=40)
    fig1.tight_layout()
    
    return fig1.figure

def predict(json_data):
    date = datetime.date.today()
    cwd = Path.cwd()
    date = datetime.date.today()
    encoder_path = f'dags/models/encoder_{date}.save'
    scaler_path = f'dags/models/scaler_{date}.save'
    encoder_path = (cwd / encoder_path).resolve()
    scaler_path = (cwd / scaler_path).resolve()
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    df, status = preprocess_new_data(json_data, encoder, scaler)

    if status != 200:
        dict = {'error': str(df.message),
                'status_code': status}
        return dict
    
    model = XGBRegressor()
    model.load_model(f'dags/models/xgbmodel_{date}.model')
    return model.predict(df)

def stream():
    src_path = build_path()
    #df = get_csv(src_path)
    #plot = make_plots(df)
    
    choice = st.sidebar.radio('Choose one to visualise',['Visuals',
                                            'Prediction'
                                            ])
    
    if choice == 'Visuals':
        st.pyplot()
    elif choice == 'Prediction':
        # JSON schema
        json_schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "integer"},
                        "property-type": {
                            "type": "string",
                            "enum": ['APARTMENT', 'HOUSE']
                        },
                        "rooms-number": {"type": "integer"},
                        "zip-code": {"type": "integer"},
                        "full-address": {"type": "string"},
                        "land-area": {"type": "integer"},
                        "garden": {"type": "boolean"},
                        "garden-area": {"type": "integer"},
                        "equipped-kitchen": {
                            "type": "string",
                            "enum": ['NOT_INSTALLED', 'USA_UNINSTALLED', 'SEMI_EQUIPPED', 'USA_SEMI_EQUIPPED', 'INSTALLED', 'USA_INSTALLED', 'HYPER_EQUIPPED', 'USA_HYPER_EQUIPPED']
                        },
                        "province": {
                            "type": "string",
                            "enum": ['Antwerp', 'Brussels', 'East Flanders', 'West Flanders', 'Flemish Brabant', 'Hainaut', 'Liège', 'Limburg', 'Luxembourg', 'Namur', 'Waloon Brabant']
                        },
                        "swimming-pool": {"type": "boolean"},
                        "furnished": {"type": "boolean"},
                        "open-fire": {"type": "boolean"},
                        "terrace": {"type": "boolean"},
                        "terrace-area": {"type": "integer"},
                        "facades-number": {"type": "integer"},
                        "building-state": {
                            "type": "string",
                            "enum": ['NEW', 'GOOD', 'RENOVATE', 'JUST RENOVATED', 'TO REBUILD']
                        },
                    },
                    "required": ["area", "property-type", "rooms-number", "zip-code"],
                    "additionalProperties": False
                }
            },
            "required": ["data"],
            "additionalProperties": False
        }
        # Create a Streamlit web app
        st.title("User Input in JSON Format")

        # Define input fields based on the JSON schema
        st.header("Property Data")

        # Define input fields based on the schema
        def create_input_field(property_name, property_info):
            if "enum" in property_info:
                selected_option = st.selectbox(property_name, property_info["enum"])
                return selected_option
            elif property_info["type"] == "integer":
                return st.number_input(property_name)
            elif property_info["type"] == "string":
                return st.text_input(property_name)
            elif property_info["type"] == "boolean":
                return st.checkbox(property_name)

        user_data = {}

        for prop_name, prop_info in json_schema["properties"]["data"]["properties"].items():
            user_data[prop_name] = create_input_field(prop_name, prop_info)

        # Convert the user data dictionary to JSON
        user_json = json.dumps({"data": user_data}, indent=4)

        # Display the JSON output
        st.subheader("Prediction")
        prediction = predict(user_json)
        st.write(prediction)

stream()