from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, distinct
from typing import List

from src.database import get_db
from src.movies import models as movie_models

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics Dashboard"]
)

# (a) For a particular movie, show the top-5 tags based on relevance scores
@router.get("/movies/{movie_id}/top-tags")
async def get_top_tags_for_movie(movie_id: int, db: AsyncSession = Depends(get_db)):
    """
    Note: The MovieLens dataset doesn't have a native 'relevance score' for tags.
    Usually, relevance is calculated by how frequently a tag is applied to a movie.
    """
    stmt = (
        select(movie_models.Tag.tag, func.count(movie_models.Tag.user_id).label("relevance_score"))
        .where(movie_models.Tag.movie_id == movie_id)
        .group_by(movie_models.Tag.tag)
        .order_by(desc("relevance_score"))
        .limit(5)
    )
    result = await db.execute(stmt)
    return [{"tag": row.tag, "relevance_score": row.relevance_score} for row in result]

# (b) Show overall and year-wise top 10 movies based on the average rating.
@router.get("/movies/top-rated")
async def get_top_rated_movies(
    year: int = Query(None, description="Filter by specific year (optional)"),
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Returns top 10 movies overall (if year is null) or for a specific year.
    Requires a minimum number of ratings to avoid 1-rating anomalies.
    """
    stmt = (
        select(
            movie_models.Movie.movie_id,
            movie_models.Movie.title,
            movie_models.Movie.year,
            func.avg(movie_models.Rating.rating).label("avg_rating"),
            func.count(movie_models.Rating.user_id).label("num_ratings")
        )
        .join(movie_models.Rating, movie_models.Movie.movie_id == movie_models.Rating.movie_id)
        .group_by(movie_models.Movie.movie_id, movie_models.Movie.title, movie_models.Movie.year)
        .having(func.count(movie_models.Rating.user_id) >= 50) # Require at least 50 ratings
    )
    
    if year:
        stmt = stmt.where(movie_models.Movie.year == year)
        
    stmt = stmt.order_by(desc("avg_rating"), desc("num_ratings")).limit(limit)
    
    result = await db.execute(stmt)
    return result.mappings().all()

# (c) What are the most popular tags given for each genre.
@router.get("/genres/top-tags")
async def get_top_tags_per_genre(db: AsyncSession = Depends(get_db)):
    """
    Finds the most popular tags for each genre.
    Uses a window function to rank tags within each genre.
    """
    # 1. Count tag frequency per genre
    subq = (
        select(
            movie_models.Genre.name.label("genre"),
            movie_models.Tag.tag.label("tag"),
            func.count(movie_models.Tag.user_id).label("tag_count")
        )
        .join(movie_models.MovieGenre, movie_models.Genre.genre_id == movie_models.MovieGenre.genre_id)
        .join(movie_models.Tag, movie_models.MovieGenre.movie_id == movie_models.Tag.movie_id)
        .group_by(movie_models.Genre.name, movie_models.Tag.tag)
        .subquery()
    )
    
    # 2. Rank them and pick top 3 for each genre
    stmt = (
        select(subq.c.genre, subq.c.tag, subq.c.tag_count)
        .distinct(subq.c.genre)
        .order_by(subq.c.genre, desc(subq.c.tag_count))
    )
    
    # Note: PostgreSQL specific DISTINCT ON syntax is handled by SQLAlchemy's distinct()
    result = await db.execute(stmt)
    return result.mappings().all()

# (d) For a particular movie, show the distribution of users' ratings.
@router.get("/movies/{movie_id}/rating-distribution")
async def get_rating_distribution(movie_id: int, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(movie_models.Rating.rating, func.count(movie_models.Rating.user_id).label("count"))
        .where(movie_models.Rating.movie_id == movie_id)
        .group_by(movie_models.Rating.rating)
        .order_by(movie_models.Rating.rating)
    )
    result = await db.execute(stmt)
    return result.mappings().all()

# (e) Show the distribution of total ratings for movies.
@router.get("/movies/rating-counts-distribution")
async def get_total_ratings_distribution(db: AsyncSession = Depends(get_db)):
    """
    Groups movies by how many ratings they have (e.g., how many movies have exactly 1 rating, 2 ratings, etc.).
    """
    stmt = (
        select(func.count(movie_models.Rating.user_id).label("num_ratings"), func.count().label("movie_count"))
        .select_from(movie_models.Rating)
        .group_by(movie_models.Rating.movie_id)
        .subquery()
    )
    
    final_stmt = (
        select(stmt.c.num_ratings, func.count(stmt.c.movie_count).label("frequency"))
        .group_by(stmt.c.num_ratings)
        .order_by(stmt.c.num_ratings)
    )
    result = await db.execute(final_stmt)
    return result.mappings().all()

# (f) For a particular year, show which genre released the highest number of movies.
@router.get("/genres/top-genre-by-year/{year}")
async def get_top_genre_by_year(year: int, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(movie_models.Genre.name, func.count(movie_models.Movie.movie_id).label("movie_count"))
        .join(movie_models.MovieGenre, movie_models.Genre.genre_id == movie_models.MovieGenre.genre_id)
        .join(movie_models.Movie, movie_models.MovieGenre.movie_id == movie_models.Movie.movie_id)
        .where(movie_models.Movie.year == year)
        .group_by(movie_models.Genre.name)
        .order_by(desc("movie_count"))
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.mappings().first()

# (g) Show genre-wise coverage of movies.
@router.get("/genres/coverage")
async def get_genre_coverage(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(movie_models.Genre.name, func.count(movie_models.MovieGenre.movie_id).label("movie_count"))
        .join(movie_models.MovieGenre, movie_models.Genre.genre_id == movie_models.MovieGenre.genre_id)
        .group_by(movie_models.Genre.name)
        .order_by(desc("movie_count"))
    )
    result = await db.execute(stmt)
    return result.mappings().all()
