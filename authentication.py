#Copyright © 2025 INNOMOTICS 
import json
import os
from flask import Blueprint, request, jsonify
from flask_bcrypt import Bcrypt
#from flask_jwt_extended import create_access_token
from config import db_name,tb_name_auth,db_hostname,db_port,tb_name_target,cvs_1,previous_Fingerprint_path
import influxdb
from multiprocessing.pool import ThreadPool
from datetime import datetime, timedelta

auth_routes = Blueprint('auth', __name__)
bcrypt = Bcrypt()

# InfluxDB Connection
DB_HOSTNAME = db_hostname
DB_PORT = db_port
DB_NAME = db_name
TB_NAME = tb_name_auth
TB_NAME_TARGET = tb_name_target

db = influxdb.InfluxDBClient(DB_HOSTNAME, DB_PORT)
db.switch_database(DB_NAME)

# **Register User**
@auth_routes.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"message": "Username and password required"}), 400

    # Check if user exists
    query = f"""SELECT * FROM {TB_NAME} WHERE username = '{username}'"""
    with ThreadPool(processes=1) as pool:
        datapoints = pool.apply_async(db.query, (query,))
        result = datapoints.get()

    if result:  # User already exists
        return jsonify({"message": "User already exists"}), 400

    # Hash Password and Store User
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    json_body = [{
        "measurement": TB_NAME,
        "fields": {
            "username": username,
            "password": hashed_password
        }
    }]
    db.write_points(json_body)

    return jsonify({"message": "User registered successfully"}), 201

# **Login User**
@auth_routes.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    # Fetch user data from InfluxDB
    query = f"""SELECT * FROM {TB_NAME} WHERE username = '{username}'"""
    with ThreadPool(processes=1) as pool:
        datapoints = pool.apply_async(db.query, (query,))
        result = datapoints.get()

    # Convert ResultSet to List
    user_data = None
    for key in result.keys():
        user_data = list(result[key])  # Get the actual records
        break  # Exit loop after first match

    if not user_data:
        return jsonify({"message": "Invalid credentials"}), 401

    stored_password = user_data[0]['password']  # Extract stored password

    # Verify password
    if not bcrypt.check_password_hash(stored_password, password):
        return jsonify({"message": "Invalid credentials"}), 401

    return jsonify({"message": "Login Successful"}), 200

def store_fingerprint_values():
    try:
        if not os.path.exists(previous_Fingerprint_path):
            print("❌ JSON file not found")
            return

        with open(previous_Fingerprint_path, 'r') as file:
            f_json_n = json.load(file)

        actions = f_json_n.get("data", [{}])[0].get("actions", {})
        row_fields = {}

        # Iterate through both control and process categories
        for category in ["control", "process"]:
            for action in actions.get(category, []):
                if "var_name" not in action:
                    continue
                var = action["var_name"]
                if var in cvs_1:
                    fp = action.get("fingerprint_set_point")
                    if fp is not None:
                        row_fields[var] = float(fp)

        # Only write if we got some fields
        if row_fields:
            point = {
                "measurement": TB_NAME_TARGET,
                "fields": row_fields
            }
            db.write_points([point])
    except Exception as e:
        print(f"❌ Error in fingerprint job: {str(e)}")
