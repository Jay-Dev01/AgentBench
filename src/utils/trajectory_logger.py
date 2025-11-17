# src/utils/trajectory_logger.py
import json
from typing import List, Dict, Any

class TrajectoryLogger:
    def __init__(self, task_id: str, task_text: str, environment: str):
        self.data: Dict[str, Any] = {
            "metadata": {
                "task_id": task_id,
                "task": task_text,
                "success": False,
                "won": False,
                "environment": environment,
            },
            "messages": []  # list of {"role": "user"/"assistant", "content": "..."}
        }

    def add_user(self, text: str):
        self.data["messages"].append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.data["messages"].append({"role": "assistant", "content": text})

    def mark_success(self, success: bool):
        self.data["metadata"]["success"] = bool(success)
        self.data["metadata"]["won"] = bool(success)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
