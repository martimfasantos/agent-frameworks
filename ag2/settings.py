import pydantic
from pydantic_settings import BaseSettings

# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    OPENAI_API_KEY: pydantic.SecretStr

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings: Settings = Settings() 
