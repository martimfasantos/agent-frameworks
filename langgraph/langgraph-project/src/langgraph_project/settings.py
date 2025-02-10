import pydantic
from pydantic_settings import BaseSettings


# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    openai_api_key: pydantic.SecretStr
    openai_model_name: str = "gpt-4o-mini"
    embeddings_model_name: str = "text-embedding-ada-002"
    knowledge_base_path: str = "./knowledge-base"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings: Settings = Settings()