import time
import json
from typing import List, Dict, Any, Optional, Union


# Constants and Type Aliases
LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]

LogLevel = Union["DEBUG", "INFO", "WARN", "ERROR"]

EVENT_TYPES = [
    "SEND_PROMPT", "RETRIEVE_HISTORY", "RETRIEVE_CONTEXT", "FINAL_PROMPT",
    "FORMAT_HISTORY", "LLM_RESPONSE", "ERROR"
]

class ChatLogEntry:
    def __init__(
        self,
        timestamp: int,
        log_level: LogLevel,
        event_type: str,
        details: Any,
        latency: Optional[int] = None
    ):
        self.timestamp = timestamp
        self.log_level = log_level
        self.event_type = event_type
        self.details = details
        self.latency = latency

    def to_dict(self):
        entry = {
            "timestamp": self.timestamp,
            "logLevel": self.log_level,
            "eventType": self.event_type,
            "details": self.details
        }
        if self.latency:
            entry["latency"] = f"{self.latency}ms"
        return entry

class ChatLoggerOptions:
    def __init__(self, log_level: LogLevel):
        self.log_level = log_level
        self.log_output = 'console'

class ChatLogger:
    def __init__(self, options: ChatLoggerOptions):
        self.logs: List[ChatLogEntry] = []
        self.options = options
        self.event_start_times: Dict[str, int] = {}
    async def log(self, level: LogLevel, event_type: str, details: Any, latency: Optional[int] = None):
        if self.should_log(level):
            timestamp = int(time.time() * 1000)
            log_entry = ChatLogEntry(timestamp, level, event_type, details, latency)
            self.logs.append(log_entry)

            if self.options.log_output == "console":
                await self.write_to_console(log_entry)
            
            time.sleep(0.1)
            
    async def write_to_console(self, log_entry: ChatLogEntry):
        JSON_SPACING = 2
        print(json.dumps(log_entry.to_dict(), indent=JSON_SPACING))

    def should_log(self, level: LogLevel) -> bool:
        level_index = LOG_LEVELS.index(level)
        option_level_index = LOG_LEVELS.index(self.options.log_level)
        return level_index >= option_level_index

    def start_timer(self, event_type: str):
        self.event_start_times[event_type] = int(time.time() * 1000)

    def end_timer(self, event_type: str) -> Optional[int]:
        start_time = self.event_start_times.pop(event_type, None)
        if start_time is not None:
            return int(time.time() * 1000) - start_time
        return None

    async def log_send_prompt(self, prompt: str):
        self.log("INFO", "SEND_PROMPT", {"prompt": prompt})

    def start_retrieve_history(self):
        self.start_timer("RETRIEVE_HISTORY")

    async def end_retrieve_history(self, history: List[Any]):
        latency = self.end_timer("RETRIEVE_HISTORY")
        self.log("INFO", "RETRIEVE_HISTORY", {"history": history}, latency)

    def start_retrieve_context(self):
        self.start_timer("RETRIEVE_CONTEXT")

    async def end_retrieve_context(self, context: Any):
        latency = self.end_timer("RETRIEVE_CONTEXT")
        self.log("INFO", "RETRIEVE_CONTEXT", {"context": context}, latency)

    async def log_retrieve_format_history(self, formatted_history: Any):
        self.log("INFO", "FORMAT_HISTORY", {"formattedHistory": formatted_history})

    async def log_final_prompt(self, prompt: Any):
        self.log("INFO", "FINAL_PROMPT", {"prompt": prompt})

    def start_llm_response(self):
        self.start_timer("LLM_RESPONSE")

    async def end_llm_response(self, response: Any):
        latency = self.end_timer("LLM_RESPONSE")
        self.log("INFO", "LLM_RESPONSE", {"response": response}, latency)

    async def   log_error(self, error: Exception):
        self.log("ERROR", "ERROR", {"message": str(error), "stack": error.__traceback__})

    def get_logs(self) -> List[Dict[str, Any]]:
        return [log_entry.to_dict() for log_entry in self.logs]
