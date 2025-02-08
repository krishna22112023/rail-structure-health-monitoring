from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text
from contextlib import contextmanager
from config import settings  # Import settings.py for database URI

# Configure the database URL
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# Set up the engine with robust connection pooling
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=10,  # Number of connections in the pool
    max_overflow=20,  # Additional connections allowed beyond pool_size
    pool_timeout=30,  # Timeout for obtaining a connection from the pool
    pool_recycle=1800,  # Recycle connections every 30 minutes
    pool_pre_ping=True  # Enable health checks for connections
)

# Configure the session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except OperationalError as e:
        # Log and handle database connection issues
        print(f"Database connection error: {e}")
        raise
    finally:
        db.close()


def check_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("Database connection is healthy.")
    except OperationalError as e:
        print(f"Database connection failed: {e}")