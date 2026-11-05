from sqlalchemy import String, Integer, Numeric, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database import Base
from datetime import datetime
from typing import List, Optional

class Movie(Base):
    __tablename__ = "movies"

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String, index=True)
    title_clean: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    genres_raw: Mapped[Optional[str]] = mapped_column(String, nullable=True)

class Rating(Base):
    __tablename__ = "ratings"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    movie_id: Mapped[int] = mapped_column(Integer, ForeignKey("movies.movie_id"), primary_key=True)
    rating: Mapped[float] = mapped_column(Numeric(2, 1))
    rated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    movie: Mapped["Movie"] = relationship()

class Tag(Base):
    __tablename__ = "tags"

    # We MUST define a primary key. We use a composite key made of all columns.
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    movie_id: Mapped[int] = mapped_column(Integer, ForeignKey("movies.movie_id"), primary_key=True)
    tag: Mapped[Optional[str]] = mapped_column(String, primary_key=True)
    tagged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)

    # Explicitly tell it to join on movie_id
    movie: Mapped["Movie"] = relationship(primaryjoin="Tag.movie_id == Movie.movie_id")

class Genre(Base):
    __tablename__ = "genres"

    genre_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)

class MovieGenre(Base):
    __tablename__ = "movie_genres"

    movie_id: Mapped[int] = mapped_column(Integer, ForeignKey("movies.movie_id"), primary_key=True)
    genre_id: Mapped[int] = mapped_column(Integer, ForeignKey("genres.genre_id"), primary_key=True)

    movie: Mapped["Movie"] = relationship()
    genre: Mapped["Genre"] = relationship()
    