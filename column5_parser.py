"""
Column 5 Parser: Nam Kashtakar Meh Ahval Field Extraction

This module parses the semi-structured cultivator information field from 
Jamabandi documents. It extracts:
- Cultivator name
- Parentage (S/o, D/o, W/o, Widow/o)
- Caste
- Residential village
- Ownership type (hereditary, non-heirs, temporary)
- Share fractions

Example Input:
    "kasht sahid v singh pisar attar singh kaum sukh sakindeh gair morosi"
    
Expected Output:
    {
        "name": "sahid v singh",
        "parentage_type": "S/o",
        "parentage_name": "attar singh",
        "caste": "sukh",
        "residence": "sakindeh",
        "ownership_type": "gair_morosi"
    }
"""

import re
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging

from models import CultivatorInfo, ParentageType, OwnershipType, ExtractionMethod

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Intermediate result from parsing operations"""
    value: Optional[str]
    start_pos: int
    end_pos: int
    confidence: float


class Column5Parser:
    """
    Parser for nam_kashtakar_meh_ahval (cultivator details) field.
    
    Uses keyword-based extraction with position tracking.
    """
    
    # === KEYWORD MAPPINGS ===
    
    # Role markers indicate the start of cultivator information
    ROLE_MARKERS = {
        'kasht': 'cultivator',
        'malik': 'owner',
        'bayaan': 'seller',
        'mushtari': 'buyer',
        'wahib': 'gifter',
        'mohoob': 'gift_receiver',
        'decree dehende': 'court_ordered_giver',
        'decree gerendaye': 'court_ordered_receiver',
    }
    
    # Parentage markers with their type mapping
    PARENTAGE_MARKERS = {
        'pisaran': ParentageType.SON_OF,      # sons of (check plural first)
        'pisar': ParentageType.SON_OF,        # son of
        'dukhtaran': ParentageType.DAUGHTER_OF,  # daughters of
        'dukhtar': ParentageType.DAUGHTER_OF,    # daughter of
        'zoja': ParentageType.WIFE_OF,        # wife of
        'byuh': ParentageType.WIDOW_OF,       # widow of
    }
    
    # Caste marker
    CASTE_MARKER = 'kaum'
    
    # Residence markers
    RESIDENCE_MARKERS = {
        'sakindeh': 'same_village',  # Resident of the same village
        'sakinandeh': 'same_village',  # Alternate spelling
        'sakin': 'other_village',    # Resident (of specified village)
    }
    
    # Ownership type markers
    OWNERSHIP_MARKERS = {
        'gair morosi': OwnershipType.NON_HEIRS,
        'gair marosi': OwnershipType.NON_HEIRS,  # Alternate spelling
        'morosi': OwnershipType.HEREDITARY,
        'marosi': OwnershipType.HEREDITARY,  # Alternate spelling
        'alati': OwnershipType.TEMPORARY,
        'alatiyan': OwnershipType.TEMPORARY,
    }
    
    # Fraction pattern for ownership shares
    FRACTION_PATTERN = re.compile(r'(\d+/\d+)')
    
    # Common noise words to filter out
    NOISE_WORDS = {'meh', 'ahval', 'v', 'va', 'aur', 'and'}
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the parser.
        
        Args:
            strict_mode: If True, raise errors on parsing failures. 
                        If False, return partial results with low confidence.
        """
        self.strict_mode = strict_mode
    
    def parse(self, text: str) -> CultivatorInfo:
        """
        Parse Column 5 text into structured CultivatorInfo.
        
        Args:
            text: Raw text from Column 5 (nam_kashtakar_meh_ahval)
            
        Returns:
            CultivatorInfo with extracted fields and confidence score
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to Column5Parser")
            return CultivatorInfo(
                name="EMPTY",
                raw_text=text,
                confidence=0.0,
                extraction_method=ExtractionMethod.HYBRID
            )
        
        # Normalize text
        normalized = self._normalize_text(text)
        logger.debug(f"Parsing: {normalized[:100]}...")
        
        # Extract components
        role = self._extract_role(normalized)
        name_result = self._extract_name(normalized)
        parentage_type, parentage_name = self._extract_parentage(normalized)
        caste = self._extract_caste(normalized)
        residence, residence_type = self._extract_residence(normalized)
        ownership_type = self._extract_ownership_type(normalized)
        share_fraction = self._extract_share_fraction(normalized)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            name_result.value, parentage_type, caste, residence, ownership_type
        )
        
        # Build result
        result = CultivatorInfo(
            name=name_result.value or "UNKNOWN",
            parentage_type=parentage_type,
            parentage_name=parentage_name,
            caste=caste,
            residence=residence,
            residence_type=residence_type,
            ownership_type=ownership_type,
            share_fraction=share_fraction,
            role=role,
            raw_text=text,
            confidence=confidence,
            extraction_method=ExtractionMethod.HYBRID
        )
        
        logger.debug(f"Parsed result: {result.model_dump()}")
        return result
    
    def parse_multiple(self, text: str) -> List[CultivatorInfo]:
        """
        Parse text that may contain multiple cultivators.
        
        Handles cases like: "iqubal singh 1/3 manmohan singh 1/3 harbans singh 1/3 pisaran bachan singh"
        
        Args:
            text: Raw text that may contain multiple cultivator entries
            
        Returns:
            List of CultivatorInfo for each cultivator found
        """
        # Check for share fractions as separator
        fractions = self.FRACTION_PATTERN.findall(text)
        
        if len(fractions) > 1:
            # Multiple cultivators with shares
            results = self._parse_multi_cultivator_text(text, fractions)
            if results:
                return results
        
        # Single cultivator
        return [self.parse(text)]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent parsing"""
        # Lowercase
        text = text.lower().strip()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common punctuation but keep forward slash for fractions
        text = re.sub(r'[,;:]', ' ', text)
        
        return text
    
    def _extract_role(self, text: str) -> str:
        """Extract role marker (kasht, malik, etc.)"""
        for marker, role in self.ROLE_MARKERS.items():
            if marker in text:
                return role
        return "cultivator"  # Default
    
    def _extract_name(self, text: str) -> ParseResult:
        """
        Extract cultivator name.
        
        Name is typically:
        - After a role marker (kasht, malik)
        - Before a parentage marker (pisar, dukhtar, zoja, byuh)
        """
        # Find role marker position
        role_end = 0
        for marker in self.ROLE_MARKERS:
            if marker in text:
                role_end = text.index(marker) + len(marker)
                break
        
        # Find parentage marker position
        parentage_start = len(text)
        for marker in self.PARENTAGE_MARKERS:
            if marker in text:
                pos = text.index(marker)
                if pos > role_end:  # Must be after role
                    parentage_start = min(parentage_start, pos)
        
        # Also check for fraction (indicates shared ownership before parentage)
        fraction_match = self.FRACTION_PATTERN.search(text[role_end:parentage_start])
        if fraction_match:
            parentage_start = role_end + fraction_match.start()
        
        # Extract name
        name = text[role_end:parentage_start].strip()
        
        # Clean noise words
        name_words = [w for w in name.split() if w not in self.NOISE_WORDS]
        name = ' '.join(name_words)
        
        # Calculate confidence based on name quality
        confidence = 0.9 if name and len(name) > 2 else 0.3
        
        return ParseResult(
            value=name if name else None,
            start_pos=role_end,
            end_pos=parentage_start,
            confidence=confidence
        )
    
    def _extract_parentage(self, text: str) -> Tuple[Optional[ParentageType], Optional[str]]:
        """
        Extract parentage type and name.
        
        Pattern: {marker} {name} (until kaum or residence marker)
        Example: "pisar attar singh" → (S/o, "attar singh")
        """
        for marker, ptype in self.PARENTAGE_MARKERS.items():
            if marker in text:
                start = text.index(marker) + len(marker)
                
                # Find end boundary (caste marker or residence marker)
                end = len(text)
                
                # Check for kaum
                if self.CASTE_MARKER in text[start:]:
                    end = min(end, start + text[start:].index(self.CASTE_MARKER))
                
                # Check for residence markers
                for res_marker in self.RESIDENCE_MARKERS:
                    if res_marker in text[start:]:
                        end = min(end, start + text[start:].index(res_marker))
                
                # Check for ownership markers
                for own_marker in self.OWNERSHIP_MARKERS:
                    if own_marker in text[start:]:
                        end = min(end, start + text[start:].index(own_marker))
                
                parentage_name = text[start:end].strip()
                
                # Clean noise words
                name_words = [w for w in parentage_name.split() if w not in self.NOISE_WORDS]
                parentage_name = ' '.join(name_words)
                
                if parentage_name:
                    return ptype, parentage_name
        
        return None, None
    
    def _extract_caste(self, text: str) -> Optional[str]:
        """
        Extract caste.
        
        Pattern: kaum {caste_name} (until residence or ownership marker)
        Example: "kaum sukh" → "sukh"
        """
        if self.CASTE_MARKER not in text:
            return None
        
        start = text.index(self.CASTE_MARKER) + len(self.CASTE_MARKER)
        end = len(text)
        
        # Find end boundary
        for marker in list(self.RESIDENCE_MARKERS.keys()) + list(self.OWNERSHIP_MARKERS.keys()):
            if marker in text[start:]:
                end = min(end, start + text[start:].index(marker))
        
        caste = text[start:end].strip()
        
        # Clean and validate
        caste_words = [w for w in caste.split() if w not in self.NOISE_WORDS and len(w) > 1]
        caste = ' '.join(caste_words)
        
        return caste if caste else None
    
    def _extract_residence(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract residence information.
        
        Pattern: sakin/sakindeh {village_name}
        Example: "sakindeh gair" → ("sakindeh", "same_village")
        """
        for marker, res_type in self.RESIDENCE_MARKERS.items():
            if marker in text:
                start = text.index(marker)
                
                # For 'sakindeh', the value is just the marker itself (same village)
                if res_type == 'same_village':
                    return marker, res_type
                
                # For 'sakin', extract the following village name
                start += len(marker)
                end = len(text)
                
                # Find end boundary
                for own_marker in self.OWNERSHIP_MARKERS:
                    if own_marker in text[start:]:
                        end = min(end, start + text[start:].index(own_marker))
                
                village = text[start:end].strip()
                return village if village else marker, res_type
        
        return None, None
    
    def _extract_ownership_type(self, text: str) -> Optional[OwnershipType]:
        """
        Extract ownership type.
        
        Checks for: morosi (hereditary), gair morosi (non-heirs), alati (temporary)
        """
        # Check in order (longer phrases first to avoid partial matches)
        for marker, own_type in sorted(self.OWNERSHIP_MARKERS.items(), key=lambda x: -len(x[0])):
            if marker in text:
                return own_type
        
        return None
    
    def _extract_share_fraction(self, text: str) -> Optional[str]:
        """
        Extract ownership share fraction.
        
        Pattern: 1/3, 1/10, 6/13, etc.
        """
        match = self.FRACTION_PATTERN.search(text)
        if match:
            return match.group(1)
        return None
    
    def _parse_multi_cultivator_text(
        self, 
        text: str, 
        fractions: List[str]
    ) -> List[CultivatorInfo]:
        """
        Parse text containing multiple cultivators with shares.
        
        Example: "iqubal singh 1/3 manmohan singh 1/3 harbans singh 1/3 pisaran bachan singh kaum namalum"
        """
        results = []
        
        # Split by fractions
        parts = self.FRACTION_PATTERN.split(text)
        
        # parts will be: [name1, fraction1, name2, fraction2, ...]
        # Last part may contain shared parentage/caste
        
        # Find shared parentage/caste from last part
        shared_parentage_type = None
        shared_parentage_name = None
        shared_caste = None
        
        last_part = parts[-1] if len(parts) % 2 == 1 else ""
        
        if last_part:
            shared_parentage_type, shared_parentage_name = self._extract_parentage(last_part)
            shared_caste = self._extract_caste(last_part)
        
        # Process each name-fraction pair
        for i in range(0, len(parts) - 1, 2):
            name = parts[i].strip()
            fraction = fractions[i // 2] if i // 2 < len(fractions) else None
            
            # Clean name
            name_words = [w for w in name.split() if w not in self.NOISE_WORDS and len(w) > 1]
            name = ' '.join(name_words)
            
            if name:
                cultivator = CultivatorInfo(
                    name=name,
                    parentage_type=shared_parentage_type,
                    parentage_name=shared_parentage_name,
                    caste=shared_caste,
                    share_fraction=fraction,
                    raw_text=text,
                    confidence=0.75,  # Slightly lower for multi-parse
                    extraction_method=ExtractionMethod.HYBRID
                )
                results.append(cultivator)
        
        return results if results else None
    
    def _calculate_confidence(
        self,
        name: Optional[str],
        parentage_type: Optional[ParentageType],
        caste: Optional[str],
        residence: Optional[str],
        ownership_type: Optional[OwnershipType]
    ) -> float:
        """
        Calculate parsing confidence based on extracted fields.
        
        Weights:
        - Name (required): 0.4
        - Parentage: 0.2
        - Caste: 0.2
        - Residence: 0.1
        - Ownership: 0.1
        """
        score = 0.0
        
        # Name is critical
        if name and len(name) > 2:
            score += 0.4
        elif name:
            score += 0.2
        
        # Parentage
        if parentage_type:
            score += 0.2
        
        # Caste
        if caste:
            score += 0.2
        
        # Residence
        if residence:
            score += 0.1
        
        # Ownership
        if ownership_type:
            score += 0.1
        
        return round(score, 2)


# === Convenience Functions ===

def parse_column5(text: str) -> CultivatorInfo:
    """Quick parse function for single cultivator text"""
    parser = Column5Parser()
    return parser.parse(text)


def parse_column5_multiple(text: str) -> List[CultivatorInfo]:
    """Quick parse function for potentially multiple cultivators"""
    parser = Column5Parser()
    return parser.parse_multiple(text)


# === Testing ===

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "kasht sahid v singh pisar attar singh kaum sukh sakindeh gair morosi",
        "kasht sandhu singh pisar sain kaum rakwal sakindeh alati",
        "iqubal singh 1/3 manmohan singh 1/3 harbans singh 1/3 pisaran bachan singh kaum namalum murusyan 3",
        "dharam singh pisar chatar singh kaum brahman sukh alati",
        "mohinder kaur 1/1 dukhter moti singh kaum namalum morosi",
        "ramaan kaur byuh gunjan singh kaum namalum",
    ]
    
    parser = Column5Parser()
    
    print("=" * 80)
    print("Column 5 Parser Test Results")
    print("=" * 80)
    
    for text in test_cases:
        print(f"\nInput: {text}")
        print("-" * 40)
        result = parser.parse(text)
        print(f"Name: {result.name}")
        print(f"Parentage: {result.parentage_type} {result.parentage_name}")
        print(f"Caste: {result.caste}")
        print(f"Residence: {result.residence}")
        print(f"Ownership: {result.ownership_type}")
        print(f"Share: {result.share_fraction}")
        print(f"Confidence: {result.confidence}")