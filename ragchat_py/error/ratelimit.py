from constants import RATELIMIT_ERROR_MESSAGE
from typing import TypedDict, Optional

class RatelimitResponse(TypedDict):
    error: str
    resetTime: Optional[int]

class RatelimitUpstashError(Exception):
    def __init__(self, message: str, cause: RatelimitResponse):
        super().__init__(message)
        self.name = "RatelimitError"
        self.cause = cause