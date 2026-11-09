from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database connection string (uses the URL-encoded password)
    DATABASE_URL: str

    # JWT Configuration
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 90

    # Tell Pydantic to look for the .env file in the current working directory
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"  # Ignores extra environment variables
    )

# Create a single instance to import across the app
settings = Settings()
