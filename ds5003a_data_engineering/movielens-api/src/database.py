from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from src.config import settings

# 1. Create the Async Engine
# echo=False prevents logging every SQL query to the console. Set to True if you want to debug.
engine = create_async_engine(settings.DATABASE_URL, echo=False)

# 2. Create the Async Session Factory
# expire_on_commit=False is crucial for FastAPI: it prevents SQLAlchemy from trying to 
# refresh objects from the database after a commit, which would fail because the session is closed.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# 3. Create the Base class for our SQLAlchemy models
# All our models (Users, Movies, etc.) will inherit from this.
class Base(DeclarativeBase):
    pass

# 4. FastAPI Dependency to get the database session
# This yields the session to the route, and automatically closes it when the request finishes.
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            # Note: We don't auto-commit here. Route functions should explicitly call `await db.commit()`
            # when making changes (e.g., inserting a new user or updating a rating).
        finally:
            await session.close()