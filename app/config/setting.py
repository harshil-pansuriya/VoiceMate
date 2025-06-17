from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    groq_api_key: str

    pinecone_api_key: str
    pinecone_index: str 
    
    model_config= SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
Config= Settings()
