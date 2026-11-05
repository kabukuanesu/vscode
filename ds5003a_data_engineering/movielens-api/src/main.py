from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.auth.router import router as auth_router
from src.analytics.router import router as analytics_router
from src.movies.router import router as movies_router

app = FastAPI(
    title="MovieLens Data Engineering API",
    description="Backend for DS5003A Data Engineering Lab Project",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
app.include_router(auth_router)
app.include_router(analytics_router)
app.include_router(movies_router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the MovieLens API. Go to /docs for the Swagger UI."
    }
