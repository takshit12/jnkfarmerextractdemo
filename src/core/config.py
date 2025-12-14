"""
Core Configuration Management for AgriStack

Centralized configuration using Pydantic Settings with environment variable support.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from pathlib import Path


class ExtractionConfig(BaseSettings):
    """PDF extraction configuration"""
    model_config = SettingsConfigDict(extra='ignore')
    
    camelot_accuracy_threshold: float = Field(75.0, description="Minimum accuracy for Camelot extraction")
    camelot_line_scale: int = Field(40, description="Line scale for table detection")
    camelot_flavor: Literal["lattice", "stream"] = Field("lattice", description="Camelot extraction mode")
    
    pdfplumber_enabled: bool = Field(True, description="Enable pdfplumber fallback")
    pdfplumber_snap_tolerance: int = Field(5, description="Snap tolerance for table detection")
    
    max_retries: int = Field(3, description="Maximum extraction retry attempts")
    timeout_seconds: int = Field(300, description="Extraction timeout in seconds")


class LLMConfig(BaseSettings):
    """LLM integration configuration"""
    model_config = SettingsConfigDict(extra='ignore')
    
    enabled: bool = Field(True, description="Enable LLM processing")
    provider: Literal["openai", "anthropic"] = Field("openai", description="LLM provider")
    model: str = Field("gpt-4o", description="Model name")
    
    max_tokens: int = Field(2000, description="Maximum tokens per request")
    temperature: float = Field(0.0, description="Temperature for generation")
    
    confidence_threshold: float = Field(0.7, description="Threshold for LLM fallback")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout_seconds: int = Field(30, description="Request timeout")
    
    # API keys from environment
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")


class ValidationConfig(BaseSettings):
    """Data validation configuration"""
    model_config = SettingsConfigDict(extra='ignore')
    
    strict_mode: bool = Field(False, description="Strict validation mode")
    confidence_high_threshold: float = Field(0.9, description="High confidence threshold")
    confidence_medium_threshold: float = Field(0.7, description="Medium confidence threshold")
    confidence_flag_threshold: float = Field(0.8, description="Flag records below this")
    
    required_fields: list[str] = Field(
        default_factory=lambda: ["number_khata", "cultivator_name"],
        description="Required fields for valid record"
    )


class OutputConfig(BaseSettings):
    """Output configuration"""
    model_config = SettingsConfigDict(extra='ignore')
    
    format: Literal["xlsx", "csv", "json"] = Field("xlsx", description="Output format")
    include_confidence: bool = Field(True, description="Include confidence scores")
    highlight_low_confidence: bool = Field(True, description="Highlight low confidence rows")
    highlight_color: str = Field("FFFF00", description="Highlight color (hex)")
    
    output_dir: Path = Field(Path("output"), description="Output directory")


class CacheConfig(BaseSettings):
    """Caching configuration"""
    model_config = SettingsConfigDict(extra='ignore')
    
    enabled: bool = Field(True, description="Enable caching")
    cache_dir: Path = Field(Path(".cache"), description="Cache directory")
    max_size_mb: int = Field(500, description="Maximum cache size in MB")
    ttl_hours: int = Field(24, description="Cache TTL in hours")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    model_config = SettingsConfigDict(extra='ignore')
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Log level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    file_enabled: bool = Field(True, description="Enable file logging")
    file_path: Path = Field(Path("logs/agristack.log"), description="Log file path")
    
    console_enabled: bool = Field(True, description="Enable console logging")
    json_enabled: bool = Field(False, description="Enable JSON logging")


class PerformanceConfig(BaseSettings):
    """Performance and scalability configuration"""
    model_config = SettingsConfigDict(extra='ignore')
    
    batch_size: int = Field(10, description="Batch processing size")
    max_workers: int = Field(4, description="Maximum parallel workers")
    
    enable_profiling: bool = Field(False, description="Enable performance profiling")
    enable_metrics: bool = Field(True, description="Enable metrics collection")


class AppConfig(BaseSettings):
    """Main application configuration"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra='ignore'  # Ignore extra fields from old settings.yaml
    )
    
    # Application metadata
    app_name: str = Field("AgriStack Land Record Digitization", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    
    # Component configurations
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "AppConfig":
        """Load configuration from YAML file"""
        import yaml
        
        with open(yaml_path, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def save_to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file"""
        import yaml
        
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False
            )


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global configuration instance"""
    global _config
    
    if _config is None:
        # Try to load from settings.yaml, fallback to defaults
        settings_path = Path("settings.yaml")
        if settings_path.exists():
            _config = AppConfig.load_from_yaml(settings_path)
        else:
            _config = AppConfig()
    
    return _config


def set_config(config: AppConfig) -> None:
    """Set global configuration instance"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults"""
    global _config
    _config = None
