"""
LLM Routing Logic

Determines when to use rule-based parsing vs LLM based on confidence.
"""

from typing import Dict, Any, Callable
from dataclasses import dataclass

from src.core.logger import get_logger
from src.core.config import get_config
from src.llm.processor import LLMProcessor

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    use_llm: bool
    reason: str
    confidence: float


class LLMRouter:
    """
    Routes between rule-based and LLM processing based on confidence
    """
    
    def __init__(self, llm_processor: LLMProcessor):
        """
        Initialize router
        
        Args:
            llm_processor: LLM processor instance
        """
        self.llm_processor = llm_processor
        self.confidence_threshold = get_config().llm.confidence_threshold
        self.llm_enabled = get_config().llm.enabled
        
        logger.info(f"LLM Router initialized (threshold={self.confidence_threshold}, enabled={self.llm_enabled})")
    
    def should_use_llm(self, confidence: float, field_name: str) -> RoutingDecision:
        """
        Determine if LLM should be used
        
        Args:
            confidence: Confidence score from rule-based parsing
            field_name: Name of field being parsed
            
        Returns:
            Routing decision
        """
        if not self.llm_enabled:
            return RoutingDecision(
                use_llm=False,
                reason="LLM disabled in configuration",
                confidence=confidence
            )
        
        if confidence >= self.confidence_threshold:
            return RoutingDecision(
                use_llm=False,
                reason=f"Rule-based confidence {confidence:.2f} above threshold {self.confidence_threshold:.2f}",
                confidence=confidence
            )
        
        return RoutingDecision(
            use_llm=True,
            reason=f"Rule-based confidence {confidence:.2f} below threshold {self.confidence_threshold:.2f}",
            confidence=confidence
        )
    
    def process_with_fallback(
        self,
        text: str,
        rule_based_parser: Callable[[str], Dict[str, Any]],
        llm_parser: Callable[[str], Dict[str, Any]],
        field_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Process text with rule-based parser, fallback to LLM if needed
        
        Args:
            text: Text to parse
            rule_based_parser: Function that returns dict with 'confidence' key
            llm_parser: Function that calls LLM
            field_name: Name of field for logging
            
        Returns:
            Parsed result with metadata
        """
        # Try rule-based first
        logger.debug(f"Attempting rule-based parsing for {field_name}")
        rule_result = rule_based_parser(text)
        
        confidence = rule_result.get('confidence', 0.0)
        decision = self.should_use_llm(confidence, field_name)
        
        if not decision.use_llm:
            logger.debug(f"Using rule-based result: {decision.reason}")
            rule_result['extraction_method'] = 'rule_based'
            return rule_result
        
        # Fallback to LLM
        logger.info(f"Falling back to LLM for {field_name}: {decision.reason}")
        
        try:
            llm_result = llm_parser(text)
            llm_result['extraction_method'] = 'llm'
            llm_result['confidence'] = 0.9  # LLM results generally high confidence
            llm_result['fallback_reason'] = decision.reason
            
            logger.debug(f"LLM parsing successful for {field_name}")
            return llm_result
            
        except Exception as e:
            logger.warning(f"LLM parsing failed for {field_name}: {e}")
            logger.info(f"Falling back to rule-based result despite low confidence")
            
            rule_result['extraction_method'] = 'rule_based_fallback'
            rule_result['llm_error'] = str(e)
            return rule_result
