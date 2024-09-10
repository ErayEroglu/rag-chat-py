import asyncio
from typing import Optional
from upstash_ratelimit import Ratelimit

class RateLimitService:
    def __init__(self, ratelimit: Optional[object] = None):
        self.ratelimit = ratelimit

    async def check_limit(self, session_id: str):
        if not self.ratelimit:
            # If no ratelimit object is provided, always allow the operation.
            return {
                "success": True,
                "limit": -1,
                "remaining": -1,
                "pending": asyncio.sleep(0),  # Equivalent to Promise.resolve()
                "reset": -1
            }

        result = await self.ratelimit.limit(session_id)

        return {
            "success": result.success,
            "remaining": result.remaining,
            "reset": result.reset,
            "limit": result.limit,
            "pending": asyncio.sleep(0),  # Equivalent to Promise.resolve()
            "reason": result.reason,
        }
