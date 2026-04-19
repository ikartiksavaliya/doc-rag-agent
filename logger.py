import json
import os
import datetime
from typing import Any, Dict

class AgentLogger:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.jsonl")

    def log(self, event_type: str, data: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# Global logger instance
LOG_DIR = os.getenv("LOG_DIR", "./logs")
agent_logger = AgentLogger(log_dir=LOG_DIR)
