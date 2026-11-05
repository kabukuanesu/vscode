from datetime import datetime
from sqlalchemy import String, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database import Base
from typing import Optional, List

class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, index=True)

    # Relationship to users (one role can have many users)
    users: Mapped[List["User"]] = relationship(back_populates="role")

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True, nullable=True)
    
    # Store the Argon2 or Bcrypt hash here (never plain text!)
    hashed_password: Mapped[str] = mapped_column(String(255))
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Foreign Key to the Role table
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"))
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationship to role
    role: Mapped["Role"] = relationship(back_populates="users")
    