import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Production configuration for Diabetes Assistant"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'diabetes_assistant_secret_key_2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server Configuration
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5001))
    
    # Ollama Configuration
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:latest')
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    
    # Database Configuration
    CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', './diabetes_memory')
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', './diabetes_model.pkl')
    SCALER_PATH = os.environ.get('SCALER_PATH', './scaler.pkl')
    DATASET_PATH = os.environ.get('DATASET_PATH', './diabetes.csv')
    
    # Security Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', './diabetes_assistant.log')
    
    # Performance Configuration
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', 3600))  # 1 hour

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Override with production-specific settings
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
