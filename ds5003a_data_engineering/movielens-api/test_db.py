import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(
        user="postgres",
        password="Qwerty1305Keys&.",  # Raw password with & and .
        database="movielens",
        host="localhost",
        port=5432
    )
    print("DB OK")
    await conn.close()

asyncio.run(main())