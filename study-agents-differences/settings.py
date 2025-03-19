import pydantic
from pydantic_settings import BaseSettings


# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    openai_api_key: pydantic.SecretStr
    openai_model_name: str = "gpt-4o-mini"
    azure_endpoint: str
    azure_deployment_name: str = "gpt-4o-mini"
    azure_api_version: str = "2024-10-21"
    azure_api_key: pydantic.SecretStr
    open_source_model_name: str = "watt-ai/watt-tool-70B"
    tavily_api_key: pydantic.SecretStr
    embeddings_model_name: str = "text-embedding-ada-002"
    embeddings_api_version: str = "2023-05-15"
    local_embeddings_model_name: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings: Settings = Settings()