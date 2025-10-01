from flask import Flask,request,current_app
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from functions import read_data, deviation, similarity_minus, similarity_plus,heat_consumption_real,load_json_with_retries
from functions import min_max_calculation, change_json, create_response_auto,find_past,get_end_datetime
import pandas as pd
from datetime import timedelta
from influxdb import DataFrameClient
from config import  cvs_1, auto_ranges,tags,tb_name_2,db_name,targets_new,units_2,time_var,previous_path
from config import timestamp,data_24hrs,resample_dataframe,historical_index_number
from config import previous_Fingerprint_path,socket_path,zero_vars
from config import control_parameters,control_parameters_unit,process_parameters,process_parameters_unit,min_max_ranges
import warnings
import json
import time
import itertools
import sys
import logging

warnings.filterwarnings('ignore')

app = Flask(__name__)
#CORS(app, resources={'Access-Control-Allow-Origin': '*'})
#CORS(app, resources={'/*': {'origins': '*'}})
CORS(app, resources={'/*': {'origins': '*', 'send_wildcard': False}})
socketio = SocketIO(app, cors_allowed_origins="*")

log=logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app.logger.setLevel(logging.ERROR)

socketio.ping_timeout = 60
socketio.ping_interval = 25
current_index = 0
#count = 0

def fingerprint():
    try:
        df_fingerprint = current_app.config.get('df_fingerprint')
        # get the previous time and to future from frontend
        previous_time=15
        future_time=15
        #store the json in file
        with open(previous_path,"r") as data:
            previous_json=json.load(data)

        # change the json as required by backend functions
        deviation_ranges=change_json(previous_json)

        #present time and end time
        end_time1=get_end_datetime()
        end_time=end_time1-timedelta(minutes=time_var)
        start_time=end_time-timedelta(minutes=previous_time)
        start_time_new=end_time-timedelta(minutes=data_24hrs)

        reccommendation_path = f'../files/logs/reccommendation_Auto_Log {end_time1.date()}.csv'

        # getting 1 minute data from influxdb
        real_data = read_data(start_time=start_time, end_time=end_time)
        real_data = real_data.set_index(timestamp).resample(resample_dataframe).first().reset_index()
        real_data = real_data.fillna(method='bfill')

        # copy hisorical data with -100 datapoints for t-30 and t+30 availability
        df_4_months=df_fingerprint[historical_index_number:-historical_index_number].copy().reset_index(drop=True)

        real_data_new = read_data(start_time=start_time_new, end_time=end_time)
        real_data_new = real_data_new.set_index(timestamp).resample(resample_dataframe).first().reset_index()
        real_data_new = real_data_new.fillna(method='bfill')

        fingerprint_past= find_past(real_data_new=real_data_new,deviation_ranges=deviation_ranges,previous_time=previous_time,future_time=future_time,end_time_new=end_time1)

        real_data_new1=real_data_new[:-60]
        #fingerprint_past=[]
        if len(fingerprint_past)>0:
            min_max=fingerprint_past
            fingerprint_response_last,count=create_response_auto(df_4_months= real_data_new1,real_data=real_data_new,df_complete=real_data_new,min_max_r=min_max,
                                                                previous_time=previous_time,future_time=future_time,
                                                                recommendation_path=reccommendation_path,t_minus_30=1,t_plus_30=1,min_max_total=1)
            fingerprint_response_all={"data":fingerprint_response_last}
        else:
            # get data after filtering with deviations percentage
            df_filtered_result,count_new=deviation(df_dev_filtered=df_4_months,real_data=real_data,deviation_ranges=deviation_ranges)
            #print(f"deviation:{df_filtered_result.shape[0]}")
            if df_filtered_result.shape[0]>3000:
                step=len(df_filtered_result)//3000
                df_filtered_result=df_filtered_result.iloc[::step][:3000]
            #getting the start time for all fingerprint timestamps in filtered data and real data
            
            fingerprint_t=pd.to_datetime(df_filtered_result[timestamp].values.tolist())
            
            #passing all the fingerprint and real start and end timestamps to similarity function to find t- similiary with 90% deviation

            similar_minus=similarity_minus(real_t=end_time1,fingerprint_t_r=fingerprint_t,real_data=real_data,df_similar_previous=df_fingerprint,previous_time=previous_time,
                                        deviation_ranges=deviation_ranges)
            #print(f"similar_Minus:{len(similar_minus)}")
            #passing the t- 95% similarity result to find t+ in condition checking
            similar_plus=similarity_plus(similar_minus=similar_minus,df_similar_future=df_4_months,future_time=future_time)
            #print(f"similar_Plus:{len(similar_plus)}")

            #check within min max given
            min_max=min_max_calculation(similar_plus=similar_plus,data_df=df_4_months,deviation_ranges=deviation_ranges)

            end_time1=pd.to_datetime(get_end_datetime())
            end_time=end_time1-timedelta(minutes=time_var)
            start_time=end_time-timedelta(minutes=1)

            present_data_new = read_data(start_time=start_time, end_time=end_time)

            if len(min_max)>0:
                try:
                    fingerprint_response_last,count=create_response_auto(df_4_months=df_4_months,real_data=real_data,df_complete=df_fingerprint,min_max_r=min_max,
                                                                        previous_time=previous_time,future_time=future_time,
                                                                        recommendation_path=reccommendation_path,t_minus_30=len(similar_minus),t_plus_30=len(similar_plus),
                                                                        min_max_total=len(min_max))
                    fingerprint_response_all={"data":fingerprint_response_last}
                except:
                    count=0
                    with open(previous_Fingerprint_path,"r") as data:
                        previous_fingerprint_json=json.load(data)
                        fingerprint_response_all=previous_fingerprint_json 
            else:
                count=0
                with open(previous_Fingerprint_path,"r") as data:
                    previous_fingerprint_json=json.load(data)
                    fingerprint_response_all=previous_fingerprint_json


        return fingerprint_response_all,count
    
    except Exception as e:
        # Handle any exceptions that might occur during execution
        print(f"Error in fingerprint_auto: {e}")
        return [], 0  # Return empty results in case of an error
    

@socketio.on('handle_get_difference')
def handle_get_difference():
    global current_index
    global count
    retry_fingerprint = False  # Sticky flag only for retrying failed fingerprint writes

    if request:
        while True:
            with open(socket_path, 'r') as f:
                socket = json.load(f)
                socket_val = socket.get("socket_stop")

            if socket_val:
                continue  # Go to top if socket stop is True

            end_time = get_end_datetime() - timedelta(minutes=time_var)
            start_time = end_time - timedelta(minutes=2)

            present_data = read_data(start_time=start_time, end_time=end_time)

            for key_new in auto_ranges.keys():
                end_time = get_end_datetime() - timedelta(minutes=time_var)
                start_time = end_time - timedelta(minutes=1)

                present_data = read_data(start_time=start_time, end_time=end_time)
                last_ten_values = present_data[key_new].tail(15).tolist()
                value_pairs = itertools.combinations(last_ten_values, 2)
                threshold = auto_ranges[key_new]["Lower"]

                exceeded = False
                for v1, v2 in value_pairs:
                    if abs(v2 - v1) >= threshold:
                        exceeded = True

                # Core fingerprint logic with retry support
                if exceeded or retry_fingerprint:
                    f_json, count = fingerprint()

                    if count == 0:
                        end_time = get_end_datetime() - timedelta(minutes=time_var)
                        start_time = end_time - timedelta(minutes=1)

                        real_heat = heat_consumption_real(real_df=present_data)

                        # Read previous fingerprint data
                        with open(previous_Fingerprint_path,"r") as data:
                            f_json_n=json.load(data)

                        for i, variable in enumerate(control_parameters):
                            f_json_n["data"][0]["actions"]["control"][i]["current_setpoint"] = present_data[variable].iat[-1].item()
                            f_json_n["data"][0]["actions"]["control"][i]["percentage"] = 100

                        # Update process parameters
                        for i, variable in enumerate(process_parameters):
                            f_json_n["data"][0]["actions"]["process"][i]["current_setpoint"] = present_data[variable].iat[-1].item()
                            f_json_n["data"][0]["actions"]["process"][i]["percentage"] = 100

                        f_json_n["data"][0]["fingerprint_Found"] = "True"
                        f_json_n["data"][0]["fingerprint_timestamp"] = f_json_n["data"][0]["current_timestamp"]
                        f_json_n["data"][0]["t_Minus_30_Total"] = 0
                        f_json_n["data"][0]["t_PLus_30_Total"] = 0
                        f_json_n["data"][0]["min_Max_total"] = 0
                        f_json_n["data"][0]["actions"]["summary"]["total_heat_consumption"] = real_heat

                        for target, unit in zip(targets_new, units_2):
                            f_json_n["data"][0]["actions"]["summary"][f"{target} ({unit})"] = float(present_data[target].iat[-1])

                        with open(previous_Fingerprint_path, "w") as data:
                            json.dump(f_json_n, data, indent=4)

                        retry_fingerprint = False  # Clear flag after success
                        break

                    else:
                        with open(previous_Fingerprint_path, "w") as p_data:
                            f_json_n = f_json
                            json.dump(f_json_n, p_data, indent=4)

                        retry_fingerprint = False  # Clear flag after success
                        break

                else:
                    real_heat = heat_consumption_real(real_df=present_data)

                    f_json_n = load_json_with_retries(previous_Fingerprint_path)
                    if f_json_n is None:
                        retry_fingerprint = True
                        print("Failed to load fingerprint JSON. Retrying...")
                        continue  # But don’t set retry_fingerprint here — this is normal path

                    for i, variable in enumerate(control_parameters):
                        f_json_n["data"][0]["actions"]["control"][i]["current_setpoint"] = present_data[variable].iat[-1].item()
                        f_json_n["data"][0]["actions"]["control"][i]["percentage"] = 100

                    # Update process parameters
                    for i, variable in enumerate(process_parameters):
                        f_json_n["data"][0]["actions"]["process"][i]["current_setpoint"] = present_data[variable].iat[-1].item()
                        f_json_n["data"][0]["actions"]["process"][i]["percentage"] = 100

                    # for target, unit in zip(targets_new, units_2):
                    #     f_json_n["data"][0]["actions"][len(cvs_1)][f'{target} ({unit})'] = present_data[target].iat[-1].item()

                    for target, unit in zip(targets_new, units_2):
                        f_json_n["data"][0]["actions"]["summary"][f"{target} ({unit})"] = float(present_data[target].iat[-1])

                    with open(previous_Fingerprint_path, "w") as f_data:
                        json.dump(f_json_n, f_data, indent=4)
                    break

            # Apply logic for zero_vars
            # Check both control and process actions
            for category in ["control"]:
                for action in f_json_n["data"][0]["actions"][category]:
                    if "var_name" in action and action["var_name"] in zero_vars:
                        if action.get("current_setpoint") in [0, 1]:
                            action["fingerprint_set_point"] = action["current_setpoint"]
                            action["percentage"] = 100

            # Prepare data for InfluxDB
            # Prepare data for InfluxDB
            timestamp1 = f_json_n['data'][0]['current_timestamp']
            data_dict = {'timestamp': timestamp1}

            # Go through control and process actions
            for category in ["control", "process"]:
                for action in f_json_n['data'][0]['actions'][category]:
                    var_name = action['var_name']
                    if var_name in tags.keys():
                        tag_name = tags.get(var_name)
                        data_dict[tag_name] = action["fingerprint_set_point"]

            # Convert to DataFrame
            df = pd.DataFrame([data_dict])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            try:
                timeseries_db_client = DataFrameClient(host='127.0.0.1', port=8086, database=db_name)
                timeseries_db_client.write_points(df, tb_name_2, batch_size=500)
            except Exception as e:
                print(f"Error writing to InfluxDB: {e}")

            emit('response', {'data': f_json_n})
            time.sleep(5)

    else:
        print("request context not available")

@app.route('/')
def index():
    return "WebSocket API is running!"

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    #emit('response', {'data': 'Connected to WebSocket'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
