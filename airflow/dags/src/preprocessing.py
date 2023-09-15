from jsonschema import validate
from jsonschema.exceptions import ValidationError
import pandas as pd
import numpy as np

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
        



        """    v = Draft202012Validator(json_schema)
        errors = sorted(v.iter_errors(json_data), key=str)
        status = 400

        return errors, status"""

    
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






 # model attributes: feature names


'''from sklearn.externals import joblib

# Save
encoder_filename = "models/encoder.save"
joblib.dump(encoder, encoder_filename)
# Load
encoder = joblib.load("models/encoder.save")









    property_types = ['APARTMENT', 'HOUSE']
    if jsonData['property_type'] not in property_types:
        raise ValueError("Invalid property type. Expected one of: %s" % property_types)
    
    provinces = ['Antwerp', 'Brussels', 'East Flanders', 
                     'West Flanders', 'Flemish Brabant', 'Hainaut',
                     'Liège', 'Limburg', 'Luxembourg', 'Namur', 'Waloon Brabant']
    if province not in provinces:
        provinces = list(filter(lambda x: x is not None, provinces))
        raise ValueError("Invalid kitchen type. Expected one of: %s" % provinces)
    
    kitchen_types = ['NOT_INSTALLED', 'USA_UNINSTALLED', 'SEMI_EQUIPPED', 
                     'USA_SEMI_EQUIPPED', 'INSTALLED', 'USA_INSTALLED',
                     'HYPER_EQUIPPED', 'USA_HYPER_EQUIPPED', None]
    if kitchen not in kitchen_types:
        kitchen_types = list(filter(lambda x: x is not None, kitchen_types))
        raise ValueError("Invalid kitchen type. Expected one of: %s" % kitchen_types)

    building_states = ['NEW', 'GOOD', 'RENOVATE', 
                       'JUST RENOVATED', 'TO REBUILD', None]
    if building_state not in building_states:
        building_states = list(filter(lambda x: x is not None, building_states))
        raise ValueError("Invalid building state. Expected one of: %s" % building_states)
    
    digit = str(int(digit)/100)

    arguments = locals()
    dict = {'data' : ''}
    
    for key, value in arguments.items():
        dict['data'][key] = value

    json_obj = json.dumps(dict)
    
    return json_obj


"""(living_area:int, 
                        property_type:str, 
                        number_rooms:int, 
                        digit:int, 
                        province:str = None,
                        surface_land:int = None,
                        garden:bool = None, 
                        garden_area:int = None,
                        kitchen:str = None,
                        terrace:bool = None, 
                        terrace_area:bool = None,
                        number_facades:int = None,
                        building_state:str = None)"""'''

