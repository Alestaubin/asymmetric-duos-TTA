from datetime import datetime

def log_event(msg):
    """Simple timestamped log printer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")