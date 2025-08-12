import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.main_agents.data_collector import send_data
from backend.main import gather_recent_payload, get_recent_activity, get_recent_calendar, get_goals, get_preferences
from backend.main_agents.activity_guesser import guess
from backend.main_agents.main_decider import decide
from backend.main_agents.preference_manager import update_preferences, load_preferences

# GPS test payload
gps_payload = {
    "coords": "37.7749, -122.4194",  # San Francisco coordinates
}

# Calendar test payload
calendar_payload = {
    "events": [
        {
            "title": "Meeting with team",
            "start_time": "2023-10-01T10:00:00Z",
            "end_time": "2023-10-01T11:00:00Z",
            "location": "Conference Room A",
            "description": "Discuss project milestones"
        },
        {
            "title": "Doctor Appointment",
            "start_time": "2023-10-02T09:30:00Z",
            "end_time": "2023-10-02T10:00:00Z",
            "location": "Downtown Medical Center",
            "description": "Annual physical check-up"
        }
    ]
}


# Screen test payload
screen_payload = {
    "sessions": [
        {
            "app_name": "Youtube",
            "package_name": "com.google.android.youtube",
            "start_time": "2023-10-01T09:00:00Z",
            "end_time": "2023-10-01T09:30:00Z",
            "duration_seconds": 30,
        },
        {
            "app_name": "Slack",
            "package_name": "com.Slack",
            "start_time": "2023-10-01T10:00:00Z",
            "end_time": "2023-10-01T11:00:00Z",
            "duration_seconds": 60,
        }
    ]
}

# User test payload
user_payload = {
    "data_json": {
        "user_id": "12345",
        "preferences": {
            "theme": "dark",
            "notifications": True,
        },
    },
}



send_data("GPS", gps_payload, debug=True)
send_data("CALENDAR", calendar_payload, debug=True)
send_data("SCREEN_USAGE", screen_payload, debug=True)
send_data("USER", user_payload, debug=True)

# Example usage of the activity guesser
feature_bundle = gather_recent_payload("stores", 60, debug=True)
result = guess(feature_bundle, api_mode=False, debug=True)
print(f"Activity Guess Result: {result}")

# Example usage of the main decider
goals = get_goals()  # Placeholder for actual goal retrieval
recent_calendar = get_recent_calendar()
prefs = get_preferences()
recent_activity = get_recent_activity()
action = decide(
    goals=goals,
    recent_calendar=recent_calendar,
    prefs=prefs,
    recent_activity=recent_activity,
    api_mode=False,
    debug=True
)
print(action)