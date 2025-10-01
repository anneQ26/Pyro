import os
import time
import json
import warnings
import threading
import pandas as pd
from flask_cors import CORS
from datetime import timedelta,datetime
from flask import request,jsonify, Blueprint, current_app
from config import cvs_1,targets_new,units_2,time_var,socket_path,previous_path, deviation_ranges_path,previous_Fingerprint_path,trend_Fingerprint_path
from config import data_24hrs,timestamp,resample_dataframe,fill_method, historical_index_number,zero_vars,trend_target_parameters
from functions import read_data,deviation,similarity_minus,similarity_plus,update_socket_after_delay,check_trend,get_end_datetime
from functions import min_max_calculation,change_json,create_response,find_past,heat_consumption_real,create_response_trend
from config import control_parameters,control_parameters_unit,process_parameters,process_parameters_unit,min_max_ranges
# import win32com.client
# import pythoncom
warnings.filterwarnings('ignore')

# app = Flask(__name__)
ai_routes = Blueprint('ai',__name__)
# CORS(app, resources={'Access-Control-Allow-Origin', '*'})


@ai_routes.route('/fingerprint', methods=['POST'])
def fingerprint():
    try:
        guidance_value=False
        df_fingerprint = current_app.config.get('df_fingerprint')
        # Open the JSON file in read+write mode
        with open(socket_path, 'r+') as f:
            socket= json.load(f)   # Load the existing JSON data
            socket["socket_stop"]= True   # Update the 'socket_stop' key
            f.seek(0)   # Move the file cursor to the beginning before writing updated data
            json.dump(socket, f, indent=4)   # Write updated JSON back to the file with proper indentation
            f.truncate()   # Truncate file to remove any leftover data from the previous content

        # Get JSON data from the request
        fingerprint_json= request.get_json()
        # Extracts 'previous_Time' and 'future_Time' from the JSON request.
        previous_time= fingerprint_json["previous_Time"]
        future_time= fingerprint_json["future_Time"]

        # Open the file in write mode and save the JSON data
        with open(previous_path, "w") as data:
            json.dump(fingerprint_json, data, indent= 4)

        # Convert JSON using change_json function
        deviation_ranges= change_json(f_Json= fingerprint_json)
        
        # Open file in write mode and dump JSON data
        with open(deviation_ranges_path, "w") as data:
            json.dump(deviation_ranges, data, indent= 4)

        #present time and end time
        end_time1= get_end_datetime()  # The end timestamp.
        end_time= end_time1 - timedelta(minutes= time_var)  # Time difference (in minutes) to adjust for GMT time zone.
        start_time= end_time - timedelta(minutes= previous_time)  # Time (in minutes) to determine start time.
        start_time_new= end_time - timedelta(minutes= data_24hrs) # Time (in minutes) for a 24-hour data range.

        # Format the file path using the end date
        reccommendation_path= f'../files/logs/reccommendation_Log {end_time1.date()}.csv'

        # Read the real-time data within the specified time range
        real_data= read_data(start_time= start_time, end_time= end_time)
        real_data= real_data.set_index(timestamp).resample(resample_dataframe).first().reset_index()  # Set timestamp as index and resample the data
        real_data= real_data.fillna(method= fill_method)  # Fill missing values using the specified method

        # copy hisorical data with -100 datapoints for t-30 and t+30 availability
        df_4_months= df_fingerprint.copy()[historical_index_number:-historical_index_number].reset_index(drop=True)

        # Identify valid keys where 'target' > -1
        # valid_keys = [key for key in trend_target_parameters if deviation_ranges[key]["target"] > -1]

        # # Extract columns and timestamps for analysis
        # col_data = {
        #     key: df_4_months[key].astype(float).round(0).tolist()
        #     for key in valid_keys
        # }
        # timestamps = df_4_months[timestamp].tolist()

        # # Process trends
        # trend_results = {
        #     key: check_trend(
        #         col_data[key], 
        #         timestamps, 
        #         deviation_ranges[key]["target"], 
        #         round(real_data.iloc[-1][key],0)
        #     )
        #     for key in valid_keys
        # }
            
        # # Print trend counts
        # # for key, trends in trend_results.items():
        # #     print(f"Total number of trends {key}: {len(trends)}")

        # # Find common trends
        # common_trends = set.intersection(*(set(trend_results[key]) for key in valid_keys)) if valid_keys else set()

        # # Extract start times from the common trends
        # start_time_all = [start_time for start_time, _ in common_trends] if common_trends else []

        # start_times= pd.to_datetime(start_time_all)
        # #print(f"total number of trends:{len(start_times)}")

        # similar_minus= similarity_minus(real_t= end_time1,fingerprint_t_r= start_times,real_data= real_data,df_similar_previous= df_4_months,previous_time= previous_time,deviation_ranges= deviation_ranges)
        # #print(f"total number of tminus:{len(similar_minus)}")

        # if len(similar_minus)>0:
        #     #print("found trend")
        #     fingerprint_response_last=create_response_trend(df_4_months= df_4_months, real_data= real_data, df_complete= df_fingerprint, min_max_r= similar_minus,
        #                                                     previous_time= previous_time, future_time= future_time,
        #                                                     reccommendation_path= reccommendation_path, t_minus_30= len(similar_minus), t_plus_30= 0,
        #                                                     min_max_total= 0, trends_list= common_trends)
        #     fingerprint_response_all={"data":fingerprint_response_last}
            
        #     with open(trend_Fingerprint_path, "w") as data_1:
        #         json.dump(fingerprint_response_all, data_1, indent=4)
        #     return fingerprint_response_all
        start_time = time.time()
        real_data_new = read_data(start_time=start_time_new , end_time=end_time)
        real_data_new = real_data_new.set_index(timestamp).resample(resample_dataframe).first().reset_index()
        real_data_new = real_data_new.fillna(method= fill_method)
        end_time = time.time()
        #print(f"database: {end_time - start_time:.4f} seconds")

        fingerprint_past= find_past(real_data_new= real_data_new, deviation_ranges= deviation_ranges, previous_time= previous_time, future_time= future_time, end_time_new= end_time1)

        real_data_new1=real_data_new[:-60]
        #fingerprint_past=[]
        if len(fingerprint_past)>0:
            min_max=fingerprint_past
            fingerprint_response_last=create_response(df_4_months=real_data_new1,real_data=real_data_new,df_complete=real_data_new,min_max_r=min_max,
                                                    previous_time=previous_time,future_time=future_time,guidance_value=guidance_value,reccommendation_path=reccommendation_path,
                                                    t_minus_30=1,t_plus_30=1,min_max_total=1)
            fingerprint_response_all={"data":fingerprint_response_last}
        else:
            # get data after filtering with deviations percentage
            start_time = time.time()
            df_filtered_result, count_new = deviation(df_dev_filtered=df_4_months, real_data=real_data, deviation_ranges=deviation_ranges)
            end_time = time.time()
            #print(f"Deviation Execution Time: {end_time - start_time:.4f} seconds")
            print(f"deviation: {df_filtered_result.shape[0]}")

            if df_filtered_result.shape[0] > 10000:
                step = len(df_filtered_result) // 10000
                df_filtered_result = df_filtered_result.iloc[::step][:10000]

            # Getting the start time for all fingerprint timestamps
            fingerprint_t = pd.to_datetime(df_filtered_result[timestamp].values.tolist())

            # Measure time for similarity_minus function
            start_time = time.time()
            similar_minus = similarity_minus(real_t=end_time1, fingerprint_t_r=fingerprint_t, real_data=real_data,
                                            df_similar_previous=df_fingerprint, previous_time=previous_time,
                                            deviation_ranges=deviation_ranges)
            end_time = time.time()
            #print(f"Similarity Minus Execution Time: {end_time - start_time:.4f} seconds")
            print(f"similar_Minus: {len(similar_minus)}")

            # Measure time for similarity_plus function
            start_time = time.time()
            similar_plus = similarity_plus(similar_minus=similar_minus, df_similar_future=df_4_months, future_time=future_time)
            end_time = time.time()
            #print(f"Similarity Plus Execution Time: {end_time - start_time:.4f} seconds")
            print(f"similar_Plus: {len(similar_plus)}")

            # Measure time for min_max_calculation function
            start_time = time.time()
            min_max = min_max_calculation(similar_plus=similar_plus, data_df=df_4_months, deviation_ranges=deviation_ranges)
            end_time = time.time()
            #print(f"Min-Max Calculation Execution Time: {end_time - start_time:.4f} seconds")

            min_max=similar_plus
            #print(f"min max:{len(min_max)}")
            #print(f"count:{count_new}")
            if len(min_max)>0:
                fingerprint_response_last=create_response(df_4_months=df_4_months,real_data=real_data,df_complete=df_fingerprint,min_max_r=min_max,previous_time=previous_time,future_time=future_time,
                                                        guidance_value=guidance_value,reccommendation_path=reccommendation_path,t_minus_30=len(similar_minus),
                                                        t_plus_30=len(similar_plus),min_max_total=len(min_max))
                fingerprint_response_all={"data":fingerprint_response_last}
            else:
                #print("inside2")
                real_heat = heat_consumption_real(real_df=real_data)

                with open(previous_Fingerprint_path,"r") as data:
                    f_json_n=json.load(data)

                for i, variable in enumerate(control_parameters):
                    f_json_n["data"][0]["actions"]["control"][i]["current_setpoint"] = real_data[variable].iat[-1].item()
                    f_json_n["data"][0]["actions"]["control"][i]["percentage"] = 100

                # Update process parameters
                for i, variable in enumerate(process_parameters):
                    f_json_n["data"][0]["actions"]["process"][i]["current_setpoint"] = real_data[variable].iat[-1].item()
                    f_json_n["data"][0]["actions"]["process"][i]["percentage"] = 100

                # for i in range(0,3):
                #     for var in cvs_1:
                #         f_json_n["data"][0]["fingerprint_future"][i][var]=["no changes"]

                f_json_n["data"][0]["fingerprint_Found"]="False"
                f_json_n["data"][0]["fingerprint_timestamp"]=f_json_n["data"][0]["current_timestamp"]
                f_json_n["data"][0]["t_Minus_30_Total"]=0
                f_json_n["data"][0]["t_PLus_30_Total"]=0
                f_json_n["data"][0]["min_Max_total"]=0
                f_json_n["data"][0]["actions"]["summary"]["total_heat_consumption"] = real_heat

                for target, unit in zip(targets_new, units_2):
                    f_json_n["data"][0]["actions"]["summary"][f"{target} ({unit})"] = float(real_data[target].iat[-1])

                fingerprint_response_all=f_json_n

                threading.Thread(target=update_socket_after_delay).start()
        
        for category in ["control"]:
                for action in fingerprint_response_all["data"][0]["actions"][category]:
                    if "var_name" in action and action["var_name"] in zero_vars:
                        if action.get("current_setpoint") in [0, 1]:
                            action["fingerprint_set_point"] = action["current_setpoint"]
                            action["percentage"] = 100

        #print("inside final")
        with open(previous_Fingerprint_path, "w") as data:
            json.dump(fingerprint_response_all, data, indent=4)
        

        return fingerprint_response_all
    except Exception as e:
        print(f"Error occurred: {e}")  # Log any errors that occur
        return {"error": str(e)}

@ai_routes.route('/reject_Fingerprint', methods=['GET'])
def reject_Fingerprint():
    with open('../files/json/socket.json','r+') as f:
            socket=json.load(f)
            socket["socket_stop"]= False
            f.seek(0)
            json.dump(socket,f,indent=4)
            f.truncate()
    response={"completed":"True"}
    return response

@ai_routes.route('/toggle', methods=['POST'])
def toggle():
    log_file = '../files/json/toggle.json'
#    global start_time
    data = request.get_json()
    status = data.get("status")  # True if ON, False if OFF

    if status:  # Toggle ON
        # pythoncom.CoInitialize()
        workbook_name = 'KilnExpert_Write_Server_working.xlsm'
        # workbook = get_excel_workbook(workbook_name)

        # if workbook is not None:
        #     excel = win32com.client.GetObject(None, "Excel.Application")
        #     excel.Visible = True
        #     worksheet = workbook.Sheets('write')
        #     excel.Application.Run(f'{workbook.Name}!StartFetching')
        # pythoncom.CoUninitialize()
        start_time = datetime.now()
        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                log_data = json.load(file)
        else:
            log_data = []

        # Append new log entry
        log_data.append({
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time":"_ _ _ _ _",
            "duration":"_ _ _ _ _",
            "status":True
        })

        with open('../files/json/coal.json','r+') as f:
            coal=json.load(f)
            coal["increase"]= True
            coal["decrease"]= True
            coal["increase_count"]=False
            coal["decrease_count"]=False 
            coal["count"]=0
            f.seek(0)
            json.dump(coal,f,indent=4)
            f.truncate()

        # Write back to JSON file
        with open(log_file, 'w') as file:
            json.dump(log_data, file, indent=4)
            
        return jsonify({
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time":"_ _ _ _ _",
            "duration":"_ _ _ _ _",
            "status":True
        })
    
    else:  # Toggle OFF
        # pythoncom.CoInitialize()
        workbook_name = 'KilnExpert_Write_Server_working.xlsm'
        # workbook = get_excel_workbook(workbook_name)

        # if workbook is not None:
        #     excel = win32com.client.GetObject(None, "Excel.Application")
        #     excel.Visible = True
        #     worksheet = workbook.Sheets('write')
        #     excel.Application.Run(f'{workbook.Name}!StopFetching')
        # pythoncom.CoUninitialize()
        with open(log_file, 'r') as file:
            log_data = json.load(file)
        latest_log = log_data[-1]
        start_time=latest_log.get("start_time")
        start_time=pd.to_datetime(start_time)
        end_time = datetime.now()
        total_seconds = (end_time - start_time).total_seconds()
        
        # Convert the total time into hours, minutes, and seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        # Format the duration based on the time components
        if hours > 0:
            if minutes > 0:
                duration = f"{int(hours)} hr {int(minutes)} min"
            else:
                duration = f"{int(hours)} hr"
        elif minutes > 0:
            if seconds > 0:
                duration = f"{int(minutes)} min {int(seconds)} sec"
            else:
                duration = f"{int(minutes)} min"
        else:
            duration = f"{int(seconds)} sec"

        # Read existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                log_data = json.load(file)
        else:
            log_data = []

        # Append new log entry
        log_data.append({
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration": duration,
            "status":False
        })

        # Write back to JSON file
        with open(log_file, 'w') as file:
            json.dump(log_data, file, indent=4)
        

        # Return response to frontend
        return jsonify({
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration": duration,
            "status":False  # Human-readable duration format
        })
    
@ai_routes.route('/latest', methods=['GET'])
def get_latest_log():
    log_file = '../files/json/toggle.json'
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            log_data = json.load(file)
        
        if log_data:  # Check if log data is not empty
            latest_log = log_data[-1]  # Get the latest entry
            return jsonify({
                "start_time": latest_log.get("start_time"),
                "end_time": latest_log.get("end_time"),
                "duration": latest_log.get("duration"),
                "status":latest_log.get("status")
            })
        else:
            return jsonify({"message": "No log entries found."}), 404
    else:
        return jsonify({"message": "Log file not found."}), 404

    