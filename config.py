import os  # For communicating with the OS in the virtual machine
from pydantic_settings import BaseSettings  # Import from pydantic-settings


def return_complete_dir(filename: str = ".env") -> str:
    """
    Uses the OS to return the correct path of the .env file
    """
    absolute_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(absolute_path)
    complete_dir = os.path.join(dir_name, filename)

    return complete_dir  


class Settings(BaseSettings):
    """
    Uses pydantic to define settings library for the project
    """
    groq_api_key: str

    class Config:
        env_file = return_complete_dir(".env")  # Call the function to get the full path


# Create an instance of the "settings" class to be imported in the notebook
settings = Settings()
