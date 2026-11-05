from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from src.database import get_db
from src.movies import models
from src.auth import models as auth_models
from src.auth.dependencies import get_current_user_or_admin, get_current_admin
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/movies",
    tags=["Movies & Ratings"]
)

# --- Pydantic Schemas for this router ---
class RatingUpdate(BaseModel):
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating between 0.5 and 5.0")

class MovieResponse(BaseModel):
    movie_id: int
    title: str
    year: int | None
    model_config = {"from_attributes": True}

# --- Endpoints ---

@router.get("/", response_model=List[MovieResponse])
async def search_movies(
    query: str = "",
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Search for movies by title. Open to everyone."""
    stmt = select(models.Movie)
    if query:
        stmt = stmt.where(models.Movie.title.ilike(f"%{query}%"))
    
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.patch("/{movie_id}/rating")
async def submit_rating(
    movie_id: int,
    payload: RatingUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: auth_models.User = Depends(get_current_user_or_admin) # Requires login
):
    """
    Add or update a rating for a movie. 
    Accessible by both 'user' and 'admin' roles.
    """
    # 1. Verify the movie exists
    movie = await db.get(models.Movie, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    # 2. Check if the user already rated this movie
    stmt = select(models.Rating).where(
        models.Rating.movie_id == movie_id,
        models.Rating.user_id == current_user.id
    )
    result = await db.execute(stmt)
    existing_rating = result.scalar_one_or_none()

    if existing_rating:
        # Update existing rating
        existing_rating.rating = payload.rating
    else:
        # Create new rating
        new_rating = models.Rating(
            movie_id=movie_id,
            user_id=current_user.id,
            rating=payload.rating,
            rated_at=func.now() # Use DB time
        )
        db.add(new_rating)

    await db.commit()
    return {"message": "Rating submitted successfully", "movie_id": movie_id, "rating": payload.rating}

@router.delete("/{movie_id}/rating/{user_id}")
async def delete_user_rating(
    movie_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: auth_models.User = Depends(get_current_admin) # ONLY Admins can do this
):
    """
    Delete a user's rating for a movie.
    Restricted to 'admin' role only.
    """
    stmt = select(models.Rating).where(
        models.Rating.movie_id == movie_id,
        models.Rating.user_id == user_id
    )
    result = await db.execute(stmt)
    rating = result.scalar_one_or_none()

    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")

    await db.delete(rating)
    await db.commit()
    
    return {"message": f"Rating deleted by Admin {current_admin.username}"}
