import time
from upstash_vector import Index

def sleep(seconds: int) -> None:
    """Sleep for the specified number of seconds."""
    time.sleep(seconds)

async def await_until_indexed(client: Index, timeout_millis: int = 10_000) -> None:
    """Wait until indexing is complete or until the timeout is reached."""
    start_time = time.time()

    async def get_info() -> dict:
        """Retrieve indexing information from the client."""
        return await client.info()

    while True:
        info = await get_info()
        if info.get("pendingVectorCount", 0) == 0:
            # OK, nothing more to index.
            return

        # Not indexed yet, sleep a bit and check again if the timeout is not passed.
        if (time.time() - start_time) * 1000 >= timeout_millis:
            break

        await sleep(1)

    raise TimeoutError(f"Indexing is not completed in {timeout_millis} ms.")
