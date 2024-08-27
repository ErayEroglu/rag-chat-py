from typing import Optional, Dict, Any
from upstash_ratelimit import RateLimit

class RateLimitService:
    def __init__(self, ratelimit: Optional[Any] = None):
        self.ratelimit = ratelimit

    async def check_limit(self, session_id: str) -> Dict[str, Any]:
        if not self.ratelimit:
            # If no ratelimit object is provided, always allow the operation.
            return {
                "success": True,
                "limit": -1,
                "remaining": -1,
                "pending": None,
                "reset": -1,
            }

        result = await self.ratelimit.limit(session_id)

        return {
            "success": result.success,
            "remaining": result.remaining,
            "reset": result.reset,
            "limit": result.limit,
            "pending": result.pending,
            "reason": result.reason,
        }