import joblib
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_apscheduler import APScheduler

# Blueprint imports
from AI import ai_routes
from auto import handle_get_difference
from expert_guidance_tool import guidance_routes
from previousInfo import previous_info_routes
from authentication import auth_routes, store_fingerprint_values  # include your scheduled job

# Config and utilities
import pandas as pd
import numpy as np
from config import fingerprint_path, timestamp,model_path
from Interactive_plot_duna import create_dash_app

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Scheduler config
class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config())

# Setup APScheduler
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Add job: run store_fingerprint_job every 30 seconds
scheduler.add_job(
    id='auto_fingerprint_saver',
    func=store_fingerprint_values,
    trigger='interval',
    seconds=30
)

# Load and prepare fingerprint DataFrame
df_fingerprint = pd.read_csv(fingerprint_path)
df_fingerprint[timestamp] = pd.to_datetime(df_fingerprint[timestamp])
for col in df_fingerprint.select_dtypes(include=[np.int64]).columns:
    df_fingerprint[col] = df_fingerprint[col].map(lambda x: int(x) if pd.notna(x) else x).astype('O')
app.config['df_fingerprint'] = df_fingerprint

#load model and save
model = joblib.load(model_path)
app.config['model'] = model

# Register Flask blueprints
app.register_blueprint(ai_routes, url_prefix="/ai")
app.register_blueprint(guidance_routes, url_prefix="/guidance")
app.register_blueprint(previous_info_routes, url_prefix="/previous")
app.register_blueprint(auth_routes, url_prefix="/auth")

# Attach Dash to Flask
dash_app = create_dash_app(app)

# Register SocketIO event
@socketio.on("get_difference")
def get_difference_handler():
    handle_get_difference()

# Start the server
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
