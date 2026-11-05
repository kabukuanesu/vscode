import asyncio
from src.database import engine, Base
# Import models so SQLAlchemy knows they exist before creating tables
from src.auth.models import Role, User 

async def init_models():
    print("Creating tables in the database...")
    async with engine.begin() as conn:
        # This tells SQLAlchemy to create any tables that don't exist yet
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully!")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_models())
    