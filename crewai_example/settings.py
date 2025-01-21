import pydantic
from pydantic_settings import BaseSettings

# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    groq_api_key: pydantic.SecretStr
    openai_api_key: pydantic.SecretStr 
    openai_model_name: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings: Settings = Settings() 
