import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_agents.data_collector import send_data

# GPS test payload
gps_payload = {
    "coords": "37.7749, -122.4194",  # San Francisco coordinates
}

send_data("GPS", gps_payload, debug=True)
