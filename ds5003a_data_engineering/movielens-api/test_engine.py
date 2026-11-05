import asyncio
from src.database import engine

async def test():
    async with engine.connect() as conn:
        print("Engine connected successfully!")
    await engine.dispose()

asyncio.run(test())
