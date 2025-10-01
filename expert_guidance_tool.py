import json
import warnings
import pandas as pd
from flask import Blueprint, current_app
from datetime import timedelta,datetime
from functions import read_data, create_response,get_end_datetime
from config import cvs_1,time_var, previous_Fingerprint_path, min_max_path, previous_guidance_Fingerprint_path, historical_index_number
warnings.filterwarnings('ignore')

guidance_routes = Blueprint('guidance',__name__)
# CORS(app, resources={'Access-Control-Allow-Origin', '*'})


@guidance_routes.route('/Guidance', methods=['GET'])
def fingerprint():
    """
    Function to generate fingerprint response based on real-time data and historical data.
    """
    try:
        df_fingerprint = current_app.config.get('df_fingerprint')

        # Define previous and future time intervals (in minutes)
        previous_time, future_time= 15, 15

        # Define present time and calculate start and end times for data retrieval
        end_time1= get_end_datetime()
        end_time= end_time1 - timedelta(minutes= time_var)
        start_time= end_time - timedelta(minutes= previous_time)

        # Path for logging recommendation guidance
        reccommendation_path= f'../files/logs/reccommendation_guidance_Log {end_time1.date()}.csv'

        # getting 1 minute data from influxdb
        real_data= read_data(start_time= start_time, end_time= end_time)

       # Copy historical data for various calculations
        df_4_months= df_fingerprint.copy()[historical_index_number:-historical_index_number].reset_index(drop= True)

        # Load previous fingerprint data
        with open(previous_Fingerprint_path, "r") as data:
                previous_fingerprint_json= json.load(data)
        similar_minus= previous_fingerprint_json["data"][0]["t_Minus_30_Total"]
        similar_plus= previous_fingerprint_json["data"][0]["t_PLus_30_Total"]
        
        # Load min-max values
        with open(min_max_path, "r") as file:
                min_max_list= json.load(file)     
        min_max= [datetime.fromisoformat(min_max_time) for min_max_time in min_max_list]
        # Generate fingerprint response if min_max values exist
        guidance_value =True
        if len(min_max) > 0:
            fingerprint_response_last= create_response(df_4_months= df_4_months, 
                                                       real_data= real_data, 
                                                       df_complete= df_fingerprint, 
                                                       min_max_r= min_max, 
                                                       previous_time= previous_time, 
                                                       future_time= future_time, 
                                                       guidance_value=guidance_value , 
                                                       reccommendation_path= reccommendation_path, 
                                                       t_minus_30= similar_minus, 
                                                       t_plus_30= similar_plus, 
                                                       min_max_total= len(min_max))
            
            fingerprint_response_all= {"data": [fingerprint_response_last[0]]}

        else:
             # Load previous guidance fingerprint data
            with open(previous_guidance_Fingerprint_path,"r") as data:
                previous_fingerprint_json= json.load(data)
                
                for i,variable in enumerate(cvs_1):
                    previous_fingerprint_json["data"][0]["actions"][i]["current_setpoint"]= real_data[variable].iat[-1] 
                
                previous_fingerprint_json["data"][0]["fingerprint_Found"]= "False"
                previous_fingerprint_json["data"][0]["t_Minus_30_Total"]= similar_minus
                previous_fingerprint_json["data"][0]["t_PLus_30_Total"]= similar_plus
                previous_fingerprint_json["data"][0]["min_Max_total"]= len(min_max)
                fingerprint_response_all= previous_fingerprint_json
        
        # Save updated guidance fingerprint data
        with open(previous_guidance_Fingerprint_path, "w") as data:
            json.dump(fingerprint_response_all, data, indent= 4)
                
        return fingerprint_response_all
    
    except Exception as e:
        return {"error": str(e)}

# if __name__ == '__main__':
#     df_fingerprint=pd.read_csv(fingerprint_path)
#     # app.run(port=8001)
    