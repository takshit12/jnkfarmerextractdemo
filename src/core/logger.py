"""
Structured Logging System for AgriStack

Provides context-aware logging with multiple output handlers and performance tracking.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager
import time

from src.core.config import get_config


class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def __init__(self):
        super().__init__()
        self.context: Dict[str, Any] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def set_context(self, **kwargs):
        """Set context for subsequent log messages"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context"""
        self.context.clear()


class JSONFormatter(logging.Formatter):
    """Format log records as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom context
        for key in dir(record):
            if not key.startswith('_') and key not in log_data:
                value = getattr(record, key)
                if isinstance(value, (str, int, float, bool, type(None))):
                    log_data[key] = value
        
        return json.dumps(log_data)


class PerformanceLogger:
    """Track and log performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: Dict[str, list[float]] = {}
    
    @contextmanager
    def track(self, operation: str):
        """Context manager to track operation duration"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(operation, duration)
            self.logger.debug(
                f"Performance: {operation} took {duration:.3f}s",
                extra={"operation": operation, "duration": duration}
            )
    
    def record_metric(self, name: str, value: float):
        """Record a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            "count": len(values),
            "total": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    
    def log_summary(self):
        """Log summary of all metrics"""
        self.logger.info("=== Performance Summary ===")
        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            self.logger.info(
                f"{name}: count={stats['count']}, "
                f"mean={stats['mean']:.3f}s, "
                f"min={stats['min']:.3f}s, "
                f"max={stats['max']:.3f}s"
            )


def setup_logging(
    name: str = "agristack",
    config_override: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Set up logging with configured handlers
    
    Args:
        name: Logger name
        config_override: Override configuration settings
        
    Returns:
        Configured logger instance
    """
    config = get_config().logging
    
    if config_override:
        for key, value in config_override.items():
            setattr(config, key, value)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.level))
    logger.handlers.clear()  # Remove existing handlers
    
    # Create formatters
    text_formatter = logging.Formatter(config.format)
    json_formatter = JSONFormatter()
    
    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level))
        console_handler.setFormatter(text_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.file_enabled:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setLevel(getattr(logging, config.level))
        file_handler.setFormatter(text_formatter)
        logger.addHandler(file_handler)
    
    # JSON handler
    if config.json_enabled:
        json_path = config.file_path.parent / f"{config.file_path.stem}.json"
        json_handler = logging.FileHandler(json_path)
        json_handler.setLevel(getattr(logging, config.level))
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)
    
    # Add context filter
    context_filter = ContextFilter()
    logger.addFilter(context_filter)
    
    # Store context filter for later use
    logger.context_filter = context_filter  # type: ignore
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logging(name)
    
    return logger


@contextmanager
def log_context(**kwargs):
    """Context manager to add context to log messages"""
    logger = logging.getLogger("agristack")
    
    if hasattr(logger, 'context_filter'):
        logger.context_filter.set_context(**kwargs)
        try:
            yield
        finally:
            logger.context_filter.clear_context()
    else:
        yield


# Create default logger
default_logger = setup_logging()
