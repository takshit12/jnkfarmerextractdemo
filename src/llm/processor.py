"""
LLM Processor for Complex Field Extraction

Handles cultivator field parsing and other complex text extraction using LLMs.
"""

import json
import time
from typing import Optional, Dict, Any, Literal
from openai import OpenAI, APIError, APITimeoutError
from anthropic import Anthropic

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.exceptions import LLMAPIError, LLMTimeoutError, LLMParsingError

logger = get_logger(__name__)


class LLMProcessor:
    """
    LLM-based processor for complex field extraction
    
    Uses OpenAI or Anthropic APIs with structured output parsing.
    """
    
    CULTIVATOR_SYSTEM_PROMPT = """You are an expert at parsing Jamabandi (land record) documents from Jammu & Kashmir.

The documents use transliterated Urdu with specific terminology:
- kasht = cultivator
- malik = owner
- pisar/pisaran = son of/sons of
- dukhtar/dukhtaran = daughter of/daughters of
- zoja = wife of
- byuh = widow of
- kaum = caste
- sakin/sakindeh = resident/resident of same village
- morosi = hereditary ownership
- gair morosi = non-hereditary ownership
- alati = temporary ownership

Parse the given text and extract structured information accurately."""
    
    def __init__(
        self,
        provider: Optional[Literal["openai", "anthropic"]] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM processor
        
        Args:
            provider: LLM provider (openai or anthropic), defaults to config
            model: Model name, defaults to config
        """
        config = get_config().llm
        
        self.provider = provider or config.provider
        self.model = model or config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.max_retries = config.max_retries
        self.timeout = config.timeout_seconds
        
        # Initialize client
        if self.provider == "openai":
            api_key = config.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            self.client = OpenAI(api_key=api_key)
            
        elif self.provider == "anthropic":
            api_key = config.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not configured")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
        
        logger.info(f"Initialized LLM processor: {self.provider}/{self.model}")
    
    def parse_cultivator_field(self, text: str) -> Dict[str, Any]:
        """
        Parse cultivator field (Column 5) using LLM
        
        Args:
            text: Raw cultivator field text
            
        Returns:
            Parsed cultivator information
            
        Raises:
            LLMAPIError: If API request fails
            LLMTimeoutError: If request times out
            LLMParsingError: If response parsing fails
        """
        prompt = f"""Parse this Jamabandi cultivator field into structured JSON:

Text: "{text}"

Extract the following fields:
1. cultivator_name: The name of the cultivator (after 'kasht' or 'malik')
2. parentage_type: One of: "S/o", "D/o", "W/o", "Widow/o" (or null if not found)
3. parentage_name: Name of father/husband (or null if not found)
4. caste: Caste name after 'kaum' (or null if not found)
5. residence: Village name after 'sakin'/'sakindeh', or "same_village" if 'sakindeh' (or null)
6. ownership_type: One of: "morosi", "gair_morosi", "alati" (or null if not found)
7. share_fraction: Ownership share if mentioned (e.g., "1/3") (or null if not found)
8. role: One of: "cultivator", "owner" (based on 'kasht' or 'malik')

Return ONLY valid JSON with these exact field names. Use null for missing fields."""
        
        return self._make_request(
            system_prompt=self.CULTIVATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            operation="parse_cultivator_field"
        )
    
    def parse_full_row(self, row_text: str) -> Dict[str, Any]:
        """
        Parse entire Jamabandi row when table extraction fails
        
        Args:
            row_text: Raw text of entire row
            
        Returns:
            Parsed row data
        """
        prompt = f"""Parse this Jamabandi land record row into structured JSON:

Raw text: "{row_text}"

Extract all available fields from the 12-column Jamabandi format:
1. number_khevat: Khevat number (integer or null)
2. number_khata: Khata number (integer or null)
3. nam_tarf_ya_patti: Nambardar name (string or null)
4. nam_malik_meh_ahval: Owner details (string or null)
5. cultivator: Parse cultivator details into sub-object with fields:
   - name, parentage_type, parentage_name, caste, residence, ownership_type, share_fraction, role
6. vasayil_abapashi: Irrigation type (string or null)
7. khasra_hal: Current survey number (string or null)
8. khasra_sabik: Previous survey number (string or null)
9. area_kanal: Area in kanal (integer or null)
10. area_marla: Area in marla (integer or null)
11. kisam_zamin: Land type (string or null)
12. havala_intakal: Mutation reference (string or null)

Return ONLY valid JSON. Use null for missing fields."""
        
        return self._make_request(
            system_prompt=self.CULTIVATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            operation="parse_full_row"
        )
    
    def _make_request(
        self,
        system_prompt: str,
        user_prompt: str,
        operation: str
    ) -> Dict[str, Any]:
        """
        Make LLM API request with retry logic
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            operation: Operation name for logging
            
        Returns:
            Parsed JSON response
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM request attempt {attempt + 1}/{self.max_retries}: {operation}")
                
                start_time = time.time()
                
                if self.provider == "openai":
                    response = self._openai_request(system_prompt, user_prompt)
                else:
                    response = self._anthropic_request(system_prompt, user_prompt)
                
                duration = time.time() - start_time
                logger.debug(f"LLM request completed in {duration:.2f}s")
                
                return response
                
            except (APIError, APITimeoutError) as e:
                logger.warning(f"LLM API error (attempt {attempt + 1}): {e}")
                
                if attempt == self.max_retries - 1:
                    raise LLMAPIError(self.provider, str(e))
                
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                raise LLMParsingError("Invalid JSON response", str(e))
        
        raise LLMAPIError(self.provider, "Max retries exceeded")
    
    def _openai_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make OpenAI API request"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
        
        content = response.choices[0].message.content
        if not content:
            raise LLMParsingError("Empty response from OpenAI", "")
        
        return json.loads(content)
    
    def _anthropic_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make Anthropic API request"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            timeout=self.timeout
        )
        
        content = response.content[0].text
        if not content:
            raise LLMParsingError("Empty response from Anthropic", "")
        
        # Extract JSON from response (Anthropic may include extra text)
        try:
            # Try to find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMParsingError(content, str(e))
