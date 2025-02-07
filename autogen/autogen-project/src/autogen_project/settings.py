import pydantic
from pydantic_settings import BaseSettings


# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    openai_api_key: pydantic.SecretStr
    openai_model_name: str = "gpt-4o-mini"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    temperature: float = 0
    max_tokens: int = 100
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    knowledge_base_path: str = "./knowledge-base"
    num_iterations: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings: Settings = Settings()