import os
from pydantic_settings import BaseSettings

def return_complete_dir(filename: str = ".env") -> str:
    """
    Return the correct path of the .env file.
    Used for local development only.
    """
    absolute_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(absolute_path)
    complete_dir = os.path.join(dir_name, filename)
    return complete_dir

class Settings(BaseSettings):
    """
    Uses pydantic to define settings.
    Loads from .env locally, but works with environment
    variables (like on Hugging Face Spaces) too.
    """
    groq_api_key: str = os.getenv("GROQ_API_KEY")  
    class Config:
        env_file = return_complete_dir(".env")  # Fallback for local use

settings = Settings()
