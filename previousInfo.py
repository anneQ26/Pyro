import json
import warnings
import pandas as pd
from flask_cors import CORS
from influxdb import DataFrameClient
from flask import Flask, jsonify, request,Blueprint
from config import tags, db_name, tb_name_2, previous_path, socket_path, previous_Fingerprint_path, cvs_1,min_max_path
from datetime import datetime
warnings.filterwarnings('ignore')

previous_info_routes= Blueprint('previous',__name__)
# CORS(app, resources= {'Access-Control-Allow-Origin', '*'})


@previous_info_routes.route('/previous', methods= ['GET'])
def previous():
    """
    Fetches the previous deviation and range data from a JSON file.
    If the file is missing or corrupt, returns a default dictionary.
    """
    try:
        # Attempt to read the JSON file
        with open(previous_path, "r") as read:
            return jsonify(json.load(read))
        
    except (FileNotFoundError, json.JSONDecodeError):
        # Handle missing file or JSON parsing errors
        previous_json= {
    "previous_Time": 15,
    "future_Time": 15,
    "deviation": {"K5027 TE Kiln channel temperature SH1": {"Lower": 90, "Higher": 110,"Min": 0,"Max": 1200,"order": 1,"target": -1},
                "K5028 TE Kiln channel temperarute SH2": {"Lower": 90, "Higher": 110,"Min": 0,"Max": 1200,"order": 2,"target": -1},
                "K5026 TE1 Kiln temperature (Pyrometer 1)": {"Lower": 90, "Higher": 110,"Min": 300,"Max": 2000,"order": 3,"target": -1}, 
                "K5026 TE2 Kiln temperature (Pyrometer 2)": {"Lower": 90, "Higher": 110,"Min": 300,"Max": 2000,"order": 4,"target": -1},
                "K5001AC Combustion Air Blower actual speed [rpm]": {"Lower": 90, "Higher": 110,"Min": 0,"Max": 2094,"order": 5,"target": -1}, 
                "K5351AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 6,"target": -1}, 
                "K5352AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 7,"target": -1},
                "K5353AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 8,"target": -1}, 
                "K5354AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 9,"target": -1},
                "K5355AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 10,"target": -1},
                "K5356AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 11,"target": -1},
                "K5357AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 12,"target": -1},
                "K5358AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 13,"target": -1},
                "K5359AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 14,"target": -1},
                "K5360AC Rotary valve speed %": {"Lower": 90,"Higher": 110,"Min": 0,"Max": 100,"order": 15,"target": -1}
                }}
        return jsonify(previous_json)


@previous_info_routes.route('/store_Previous', methods= ['POST'])
def store_previous():
    """
    Updates socket settings, stores previous fingerprint data, extracts actions, 
    processes data, and writes to InfluxDB.
    """
    try:
        # Update socket.json to set socket_stop to False
        with open(socket_path, 'r+') as f:
            socket= json.load(f)
            socket["socket_stop"]= False
            f.seek(0)
            json.dump(socket, f, indent= 4)
            f.truncate()
        
        # Get the previous fingerprint data from the request
        previous_data= request.get_json()
        
        min_max_auto_str=[previous_data["data"][0]["fingerprint_timestamp"]]
        similar_plus_fingerprint_str=[timestamp_new.isoformat() if isinstance(timestamp_new,datetime) else datetime.strptime(timestamp_new,"%Y-%m-%d %H:%M:%S").isoformat() for timestamp_new in min_max_auto_str]
        
        # Write the new min/max timestamp to min_max.json
        with open(min_max_path, "w") as data:
            json.dump(similar_plus_fingerprint_str, data)
        # Store the JSON data into a file
        with open(previous_Fingerprint_path, "w") as data:
            json.dump(previous_data, data, indent= 4)
        # Extract actions and timestamp
        actions= previous_data['data'][0]['actions'][:len(cvs_1)]
        timestamp= previous_data['data'][0]['current_timestamp']

        # Create a dictionary with tag names as keys and fingerprint_set_point as values
        data_dict= {'timestamp': timestamp}  # Start with timestamp
        
        for action in actions:
                var_name= action['var_name']
                if var_name in tags: 
                    tag_name= tags[var_name]
                    data_dict[tag_name]= action['fingerprint_set_point']

        # Convert dictionary to DataFrame
        df= pd.DataFrame([data_dict])
        df['timestamp']= pd.to_datetime(df['timestamp']) # Convert timestamp to datetime
        df.set_index('timestamp', inplace= True) # Set timestamp as index
        df= df.astype(float)  # Convert all columns to float
        
        # Write to InfluxDB
        timeseries_db_client= DataFrameClient(host='127.0.0.1', port= 8086, database=db_name)
        timeseries_db_client.write_points(df, tb_name_2, batch_size= 500)
        #print("written data into influxdb")
        return {"result":"success"}
    
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        return {"result": "failure", "error": str(e)}

# if __name__ == '__main__':
#     df_fingerprint=pd.read_csv(fingerprint_path)
    # app.run(port=8000)
    