import requests
import json

url = 'http://127.0.0.1:8000/predict'

data = {
    "data": {
        "area": 150,
        "property-type":"HOUSE",
        "rooms-number": 3,
        "zip-code": 9000,
        "land-area": 200,
        "garden": 1,
        "garden-area": 50,
        "equipped-kitchen": "NOT_INSTALLED",
        "terrace": 1,
        "terrace-area": 20,
        "facades-number": 4,
        "building-state": "NEW",
        "province": "East Flanders"
  }
}

response = requests.post(url, json=data)
print(response.text)
