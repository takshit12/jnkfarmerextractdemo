"""
Custom Exception Hierarchy for AgriStack

Provides structured error handling with context preservation.
"""

from typing import Optional, Dict, Any


class AgriStackError(Exception):
    """Base exception for all AgriStack errors"""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


# === Extraction Errors ===

class ExtractionError(AgriStackError):
    """Base class for extraction errors"""
    pass


class PDFLoadError(ExtractionError):
    """Error loading PDF file"""
    
    def __init__(self, pdf_path: str, reason: str):
        super().__init__(
            f"Failed to load PDF: {reason}",
            context={"pdf_path": pdf_path, "reason": reason},
            recoverable=False
        )


class TableExtractionError(ExtractionError):
    """Error extracting tables from PDF"""
    
    def __init__(self, page: int, reason: str):
        super().__init__(
            f"Failed to extract table from page {page}: {reason}",
            context={"page": page, "reason": reason},
            recoverable=True
        )


class LowAccuracyError(ExtractionError):
    """Extraction accuracy below threshold"""
    
    def __init__(self, page: int, accuracy: float, threshold: float):
        super().__init__(
            f"Extraction accuracy {accuracy:.1f}% below threshold {threshold:.1f}%",
            context={"page": page, "accuracy": accuracy, "threshold": threshold},
            recoverable=True
        )


# === Parsing Errors ===

class ParsingError(AgriStackError):
    """Base class for parsing errors"""
    pass


class Column5ParsingError(ParsingError):
    """Error parsing Column 5 (cultivator field)"""
    
    def __init__(self, text: str, reason: str):
        super().__init__(
            f"Failed to parse cultivator field: {reason}",
            context={"text": text[:100], "reason": reason},
            recoverable=True
        )


class KhasraSplittingError(ParsingError):
    """Error splitting Khasra numbers"""
    
    def __init__(self, khasra_text: str, reason: str):
        super().__init__(
            f"Failed to split Khasra numbers: {reason}",
            context={"khasra_text": khasra_text, "reason": reason},
            recoverable=True
        )


class MetadataExtractionError(ParsingError):
    """Error extracting metadata from PDF"""
    
    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Failed to extract metadata field '{field}': {reason}",
            context={"field": field, "reason": reason},
            recoverable=True
        )


# === Validation Errors ===

class ValidationError(AgriStackError):
    """Base class for validation errors"""
    pass


class FieldValidationError(ValidationError):
    """Field validation failed"""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation failed for field '{field}': {reason}",
            context={"field": field, "value": str(value), "reason": reason},
            recoverable=False
        )


class RecordValidationError(ValidationError):
    """Record validation failed"""
    
    def __init__(self, record_id: str, errors: list[str]):
        super().__init__(
            f"Record validation failed with {len(errors)} error(s)",
            context={"record_id": record_id, "errors": errors},
            recoverable=False
        )


class LowConfidenceError(ValidationError):
    """Record confidence below threshold"""
    
    def __init__(self, record_id: str, confidence: float, threshold: float):
        super().__init__(
            f"Record confidence {confidence:.2f} below threshold {threshold:.2f}",
            context={"record_id": record_id, "confidence": confidence, "threshold": threshold},
            recoverable=True
        )


# === LLM Errors ===

class LLMError(AgriStackError):
    """Base class for LLM errors"""
    pass


class LLMAPIError(LLMError):
    """LLM API request failed"""
    
    def __init__(self, provider: str, reason: str):
        super().__init__(
            f"LLM API request failed ({provider}): {reason}",
            context={"provider": provider, "reason": reason},
            recoverable=True
        )


class LLMTimeoutError(LLMError):
    """LLM request timed out"""
    
    def __init__(self, timeout: int):
        super().__init__(
            f"LLM request timed out after {timeout}s",
            context={"timeout": timeout},
            recoverable=True
        )


class LLMParsingError(LLMError):
    """Failed to parse LLM response"""
    
    def __init__(self, response: str, reason: str):
        super().__init__(
            f"Failed to parse LLM response: {reason}",
            context={"response": response[:200], "reason": reason},
            recoverable=True
        )


# === Export Errors ===

class ExportError(AgriStackError):
    """Base class for export errors"""
    pass


class ExcelExportError(ExportError):
    """Error exporting to Excel"""
    
    def __init__(self, output_path: str, reason: str):
        super().__init__(
            f"Failed to export to Excel: {reason}",
            context={"output_path": output_path, "reason": reason},
            recoverable=False
        )


# === Configuration Errors ===

class ConfigurationError(AgriStackError):
    """Configuration error"""
    
    def __init__(self, setting: str, reason: str):
        super().__init__(
            f"Configuration error for '{setting}': {reason}",
            context={"setting": setting, "reason": reason},
            recoverable=False
        )


# === Dependency Errors ===

class DependencyError(AgriStackError):
    """Missing or incompatible dependency"""
    
    def __init__(self, dependency: str, reason: str):
        super().__init__(
            f"Dependency error ({dependency}): {reason}",
            context={"dependency": dependency, "reason": reason},
            recoverable=False
        )
