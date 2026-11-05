from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt
from pwdlib import PasswordHash
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from src.config import settings
from src.auth import models, schemas

# Initialize the password hasher (uses Argon2 by default)
password_hash = PasswordHash.recommended()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if the provided password matches the hash."""
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a new password."""
    return password_hash.hash(password)

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[models.User]:
    """Fetch a user from the DB by username, including their role."""
    result = await db.execute(
        select(models.User)
        .options(selectinload(models.User.role)) # Eager load role to avoid extra DB hits
        .where(models.User.username == username)
    )
    return result.scalar_one_or_none()

async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[models.User]:
    """Verify username and password. Returns the user if successful, else None."""
    user = await get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def create_user(db: AsyncSession, user: schemas.UserCreate) -> models.User:
    """Create a new user in the database."""
    # 1. Hash the password
    hashed_password = get_password_hash(user.password)
    
    # 2. Find the role (default to 'user' if not found, though ideally it exists)
    result = await db.execute(select(models.Role).where(models.Role.name == user.role_name))
    role = result.scalar_one_or_none()
    if not role:
        # Fallback to 'user' if the provided role doesn't exist
        result = await db.execute(select(models.Role).where(models.Role.name == "user"))
        role = result.scalar_one()

    # 3. Create the DB model instance
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        role_id=role.id
    )
    
    # 4. Save to database
    db.add(db_user)
    await db.commit()
    
    # 5. Refresh to get the auto-generated ID and timestamps
    await db.refresh(db_user)
    
    # Re-fetch to ensure the role relationship is loaded for the response
    return await get_user_by_username(db, db_user.username)
