from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_db
from src.auth import models, service
from src.auth.schemas import TokenData

# This tells FastAPI where the login endpoint is (so Swagger UI can show the "Authorize" button)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> models.User:
    """
    Decodes the JWT token and fetches the user from the database.
    If the token is invalid or the user doesn't exist, it throws a 401 Unauthorized.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the token using our secret key
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    # Check if the user still exists in the database
    user = await service.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
        
    return user

async def get_current_active_user(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    """Ensures the user account is not disabled."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

class RoleChecker:
    """
    Dependency factory to check if the current user has a specific role.
    Usage: Depends(RoleChecker(["admin"]))
    """
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: models.User = Depends(get_current_active_user)) -> models.User:
        # The user's role is eager-loaded in the service layer, so this is fast
        if user.role.name not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Requires one of roles: {self.allowed_roles}"
            )
        return user

# Pre-built dependencies for easy use in routers
get_current_admin = RoleChecker(["admin"])
get_current_user_or_admin = RoleChecker(["user", "admin"])
