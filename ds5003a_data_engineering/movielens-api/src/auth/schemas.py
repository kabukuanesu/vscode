from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional
from datetime import datetime

# --- 1. Token Schemas (For Login Responses) ---

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- 2. Role Schemas ---

class RoleBase(BaseModel):
    name: str

class RoleResponse(RoleBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

# --- 3. User Schemas (For Registration & Responses) ---

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: Optional[EmailStr] = Field(None, description="Valid email address")

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters")
    role_name: str = Field("user", description="Role name: 'user' or 'admin'")

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    role: RoleResponse # Nested role details

    # This tells Pydantic to read data from SQLAlchemy models (orm_mode in v1)
    model_config = ConfigDict(from_attributes=True) 
    