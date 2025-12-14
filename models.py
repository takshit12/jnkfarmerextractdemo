"""
Pydantic Data Models for Jamabandi Land Records

This module defines the data structures for:
- Parsed cultivator information (Column 5)
- Khasra (survey) numbers
- Area measurements
- Complete Jamabandi records
- Village-level aggregations
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
from enum import Enum
from datetime import datetime


class ParentageType(str, Enum):
    """Type of relationship in parentage field"""
    SON_OF = "S/o"
    DAUGHTER_OF = "D/o"
    WIFE_OF = "W/o"
    WIDOW_OF = "Widow/o"


class OwnershipType(str, Enum):
    """Type of land ownership"""
    HEREDITARY = "morosi"
    NON_HEIRS = "gair_morosi"
    TEMPORARY = "alati"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Method used for extraction"""
    CAMELOT = "camelot"
    PDFPLUMBER = "pdfplumber"
    LLM = "llm"
    HYBRID = "hybrid"


class CultivatorInfo(BaseModel):
    """
    Parsed cultivator information from Column 5 (nam_kashtakar_meh_ahval)
    
    Example input: "kasht sahid v singh pisar attar singh kaum sukh sakindeh gair morosi"
    """
    name: str = Field(..., description="Cultivator name (after 'kasht' marker)")
    parentage_type: Optional[ParentageType] = Field(None, description="S/o, D/o, W/o, or Widow/o")
    parentage_name: Optional[str] = Field(None, description="Name of father/husband")
    caste: Optional[str] = Field(None, description="Caste (after 'kaum' marker)")
    residence: Optional[str] = Field(None, description="Residential village")
    residence_type: Optional[str] = Field(None, description="'sakin' or 'sakindeh'")
    ownership_type: Optional[OwnershipType] = Field(None, description="morosi, gair_morosi, or alati")
    share_fraction: Optional[str] = Field(None, description="Ownership share if mentioned (e.g., '1/3')")
    role: str = Field(default="cultivator", description="Role marker found (kasht, malik, etc.)")
    raw_text: Optional[str] = Field(None, description="Original unparsed text")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Parsing confidence score")
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.HYBRID)
    
    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Cultivator name cannot be empty')
        return v.strip()
    
    @field_validator('caste', 'residence', 'parentage_name', mode='before')
    @classmethod
    def clean_string_fields(cls, v):
        if v is None:
            return None
        return v.strip() if isinstance(v, str) else v
    
    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary for DataFrame export"""
        parentage = f"{self.parentage_type.value} {self.parentage_name}" if self.parentage_type and self.parentage_name else ""
        return {
            'cultivator_name': self.name,
            'cultivator_parentage': parentage,
            'cultivator_caste': self.caste or "",
            'cultivator_village': self.residence or "",
            'cultivator_remarks': f"{self.ownership_type.value if self.ownership_type else ''} {self.share_fraction or ''}".strip(),
        }


class KhasraInfo(BaseModel):
    """
    Khasra (survey) number information from Column 7
    
    Format: hal (current) or hal/sabik (current/previous)
    Example: "156" or "362/144"
    """
    hal: str = Field(..., description="Current survey number")
    sabik: Optional[str] = Field(None, description="Previous survey number")
    
    @classmethod
    def from_string(cls, text: str) -> "KhasraInfo":
        """Parse khasra from string like '156' or '362/144'"""
        import re
        match = re.match(r'(\d+)(?:/(\d+))?', text.strip())
        if match:
            return cls(hal=match.group(1), sabik=match.group(2))
        return cls(hal=text.strip())


class AreaInfo(BaseModel):
    """
    Land area information from Column 8
    
    Units: Kanal (20 marla = 1 kanal), Marla
    """
    kanal: int = Field(default=0, ge=0, description="Area in Kanal")
    marla: int = Field(default=0, ge=0, le=20, description="Area in Marla (max 20)")
    kisam_zamin: Optional[str] = Field(None, description="Land type (nahri, gora nahri, etc.)")
    
    @property
    def total_marla(self) -> int:
        """Convert to total marla"""
        return (self.kanal * 20) + self.marla
    
    def __str__(self) -> str:
        return f"{self.kanal}K {self.marla}M ({self.kisam_zamin or 'unknown'})"


class MutationInfo(BaseModel):
    """
    Mutation (transaction) information from Column 11
    
    Types: hembba nama (gift deed), varasat (inheritance), etc.
    """
    kisam: Optional[str] = Field(None, description="Type of mutation")
    number: Optional[str] = Field(None, description="Mutation reference number")
    raw_text: Optional[str] = Field(None, description="Original text")


class JamabandiRecord(BaseModel):
    """
    Complete Jamabandi record after parsing and transformation
    
    Represents a single row in the output Excel file.
    """
    # === Identifiers ===
    number_khevat: int = Field(..., description="Khevat (subdivision) number")
    number_khata: int = Field(..., description="Khata (account) number")
    
    # === Names ===
    nam_tarf_ya_patti: Optional[str] = Field(None, description="Nambardar name")
    nam_malik_meh_ahval: Optional[str] = Field(None, description="Historical owner details")
    
    # === Cultivator (parsed Column 5) ===
    cultivator: CultivatorInfo
    
    # === Land Details ===
    vasayil_abapashi: Optional[str] = Field(None, description="Irrigation means")
    khasra: KhasraInfo
    area: AreaInfo
    
    # === Transaction Details ===
    lagan_details: Optional[str] = Field(None, description="Tax details")
    mutalba_details: Optional[str] = Field(None, description="Cess levy")
    mutation: Optional[MutationInfo] = Field(None, description="Mutation reference")
    kaifiyat: Optional[str] = Field(None, description="Remarks")
    
    # === Metadata ===
    page_number: int = Field(..., description="Source page number")
    row_index: int = Field(default=0, description="Row index on page")
    is_sum_row: bool = Field(default=False, description="Whether this is a 'kitta' sum row")
    is_split_row: bool = Field(default=False, description="Whether this row was split from multi-khasra")
    split_index: Optional[int] = Field(None, description="Index within split group")
    extraction_confidence: float = Field(default=0.0, ge=0, le=1)
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.HYBRID)
    needs_review: bool = Field(default=False, description="Flag for manual review")
    
    @model_validator(mode='after')
    def set_review_flag(self):
        """Auto-set needs_review based on confidence"""
        if self.extraction_confidence < 0.8:
            self.needs_review = True
        return self
    
    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary for DataFrame/Excel export"""
        cultivator_flat = self.cultivator.to_flat_dict()
        
        return {
            'number_khevat': self.number_khevat,
            'number_khata': self.number_khata,
            'nam_tarf_ya_patti': self.nam_tarf_ya_patti or "",
            'nam_malik_meh_ahval': self.nam_malik_meh_ahval or "",
            **cultivator_flat,
            'vasayil_abapashi': self.vasayil_abapashi or "",
            'khasra_hal': self.khasra.hal,
            'khasra_sabik': self.khasra.sabik or "",
            'area_kanal': self.area.kanal,
            'area_marla': self.area.marla,
            'kisam_zamin': self.area.kisam_zamin or "",
            'lagan_details': self.lagan_details or "",
            'mutalba_details': self.mutalba_details or "",
            'havala_intakal': f"{self.mutation.kisam or ''} {self.mutation.number or ''}".strip() if self.mutation else "",
            'kaifiyat': self.kaifiyat or "",
            'extraction_confidence': round(self.extraction_confidence, 3),
            'needs_review': self.needs_review,
            'page_number': self.page_number,
        }


class VillageMetadata(BaseModel):
    """Header information extracted from Jamabandi document"""
    sal: str = Field(..., description="Year (e.g., '2017-18')")
    zla: str = Field(..., description="District (e.g., 'Jammu')")
    thsil: str = Field(..., description="Tehsil (e.g., 'Jammu West')")
    babata_mouza: str = Field(..., description="Village name (e.g., 'Gujral')")


class VillageJamabandi(BaseModel):
    """
    Complete Jamabandi extraction for a village
    
    This is the top-level output structure containing:
    - Village metadata (from document header)
    - All extracted records
    - Processing statistics
    """
    # === Village Info ===
    metadata: VillageMetadata
    
    # === Records ===
    records: List[JamabandiRecord] = Field(default_factory=list)
    
    # === Processing Stats ===
    source_file: str = Field(..., description="Source PDF filename")
    total_pages: int = Field(default=0)
    processed_at: datetime = Field(default_factory=datetime.now)
    
    # === Computed Statistics ===
    @property
    def total_records(self) -> int:
        return len(self.records)
    
    @property
    def average_confidence(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.extraction_confidence for r in self.records) / len(self.records)
    
    @property
    def low_confidence_count(self) -> int:
        return sum(1 for r in self.records if r.needs_review)
    
    @property
    def unique_khata_count(self) -> int:
        return len(set(r.number_khata for r in self.records))
    
    @property
    def unique_khasra_count(self) -> int:
        return len(set(r.khasra.hal for r in self.records))
    
    def get_summary(self) -> dict:
        """Get processing summary"""
        return {
            'village': self.metadata.babata_mouza,
            'district': self.metadata.zla,
            'tehsil': self.metadata.thsil,
            'year': self.metadata.sal,
            'total_records': self.total_records,
            'total_pages': self.total_pages,
            'unique_khata': self.unique_khata_count,
            'unique_khasra': self.unique_khasra_count,
            'average_confidence': round(self.average_confidence, 3),
            'records_needing_review': self.low_confidence_count,
            'review_percentage': round(self.low_confidence_count / max(self.total_records, 1) * 100, 1),
        }


class ExtractionResult(BaseModel):
    """Result of table extraction from a single page"""
    page_number: int
    dataframe_json: str  # JSON serialized DataFrame
    accuracy: Optional[float] = None
    extraction_method: ExtractionMethod
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None


# === Factory Functions ===

def create_empty_cultivator(raw_text: str = "") -> CultivatorInfo:
    """Create an empty cultivator record for failed parsing"""
    return CultivatorInfo(
        name="PARSING_FAILED",
        raw_text=raw_text,
        confidence=0.0,
        extraction_method=ExtractionMethod.HYBRID
    )


def create_empty_record(
    page_number: int,
    number_khevat: int = 0,
    number_khata: int = 0
) -> JamabandiRecord:
    """Create an empty record for failed extraction"""
    return JamabandiRecord(
        number_khevat=number_khevat,
        number_khata=number_khata,
        cultivator=create_empty_cultivator(),
        khasra=KhasraInfo(hal="0"),
        area=AreaInfo(),
        page_number=page_number,
        extraction_confidence=0.0,
        needs_review=True
    )