# Development Plan: J&K AgriStack Land Record Digitization

## 📅 Timeline Overview

**Total Duration:** 2 days (December 13-15, 2025)  
**Submission Deadline:** December 15, 2025, 11:59 PM EST

### Day 1 (December 13): Core Pipeline Development
- [ ] Hour 0-2: Project setup, dependencies, sample data preparation
- [ ] Hour 2-4: PDF extraction layer (Camelot + pdfplumber)
- [ ] Hour 4-6: Jamabandi-specific parsing (Column 5, Khasra splitting)
- [ ] Hour 6-8: LLM integration for complex cells

### Day 2 (December 14): Integration, Validation & Documentation
- [ ] Hour 0-2: Data validation and schema conformance
- [ ] Hour 2-4: Excel export and batch processing
- [ ] Hour 4-6: Testing on multiple sample documents
- [ ] Hour 6-8: Documentation, policy brief drafting

### Day 3 (December 15): Finalization
- [ ] Hour 0-4: Bug fixes, edge case handling
- [ ] Hour 4-6: Policy brief completion
- [ ] Hour 6-8: Final submission preparation

---

## 🔨 Phase 1: Project Setup (2 hours)

### 1.1 Environment Setup

```bash
# Create project directory
mkdir jk-agristack-landrecords
cd jk-agristack-landrecords

# Initialize virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install camelot-py[cv] pdfplumber pymupdf pandas openpyxl
pip install openai anthropic python-dotenv pydantic
pip install streamlit plotly  # For demo UI
pip install pytest  # For testing
```

### 1.2 Sample Data Preparation

1. Copy transliterated PDF to `data/input/`
2. Create ground truth for 2-3 pages (manual extraction for validation)
3. Document any edge cases observed

### 1.3 Configuration Setup

Create `config/settings.yaml`:
```yaml
extraction:
  camelot_accuracy_threshold: 75
  pdfplumber_fallback: true
  llm_enabled: true

llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4o"
  max_tokens: 2000
  temperature: 0

output:
  format: "xlsx"
  include_confidence_scores: true
  flag_low_confidence: true
  confidence_threshold: 0.8
```

---

## 🔨 Phase 2: PDF Extraction Layer (2 hours)

### 2.1 PDF Loader Module

**File:** `src/extractors/pdf_loader.py`

**Functionality:**
- Load PDF using PyMuPDF for metadata and page info
- Extract header information (sal, zla, thsil, babata mouza)
- Detect if document is text-based or scanned
- Return page-by-page iterator

**Key Implementation:**
```python
class JamabandiPDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.metadata = self._extract_metadata()
    
    def _extract_metadata(self) -> dict:
        """Extract village, tehsil, district from header"""
        first_page_text = self.doc[0].get_text()
        # Parse header pattern: "2017-18 : sal ... Jammu :zla ... :thsil ... :babata mouza"
        pass
    
    def get_pages(self) -> Iterator[Page]:
        """Yield pages for processing"""
        for page_num in range(len(self.doc)):
            yield JamabandiPage(self.doc[page_num], page_num)
```

### 2.2 Camelot Extractor

**File:** `src/extractors/camelot_extractor.py`

**Functionality:**
- Extract tables using Camelot Lattice mode
- Return DataFrame with accuracy score
- Handle multi-page tables

**Key Implementation:**
```python
class CamelotExtractor:
    def __init__(self, accuracy_threshold: float = 75.0):
        self.accuracy_threshold = accuracy_threshold
    
    def extract_tables(self, pdf_path: str, pages: str = "all") -> List[ExtractedTable]:
        tables = camelot.read_pdf(
            pdf_path,
            pages=pages,
            flavor="lattice",
            line_scale=40,  # Adjust for Jamabandi line thickness
            copy_text=['v'],  # Copy text vertically in spanning cells
        )
        
        results = []
        for table in tables:
            results.append(ExtractedTable(
                dataframe=table.df,
                accuracy=table.accuracy,
                page=table.page,
                extraction_method="camelot_lattice"
            ))
        return results
```

### 2.3 pdfplumber Fallback

**File:** `src/extractors/pdfplumber_extractor.py`

**Functionality:**
- Fallback for tables where Camelot fails
- Custom table settings for Jamabandi layout
- Character-level extraction for debugging

**Key Implementation:**
```python
class PdfplumberExtractor:
    def __init__(self):
        # Jamabandi-specific table settings
        self.table_settings = {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "text",
            "explicit_vertical_lines": self._get_jamabandi_columns(),
            "snap_tolerance": 5,
            "join_tolerance": 3,
        }
    
    def _get_jamabandi_columns(self) -> List[float]:
        """Return x-coordinates for 12 Jamabandi columns"""
        # These need to be calibrated from sample documents
        # Columns are right-to-left: 1,2,3...12
        return [50, 100, 150, 220, 350, 420, 480, 520, 560, 600, 680, 750]
    
    def extract_tables(self, pdf_path: str) -> List[ExtractedTable]:
        with pdfplumber.open(pdf_path) as pdf:
            results = []
            for page in pdf.pages:
                table = page.extract_table(self.table_settings)
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    results.append(ExtractedTable(
                        dataframe=df,
                        accuracy=None,  # pdfplumber doesn't provide accuracy
                        page=page.page_number,
                        extraction_method="pdfplumber"
                    ))
            return results
```

---

## 🔨 Phase 3: Jamabandi-Specific Parsing (2 hours)

### 3.1 Column 5 Parser (Nam Kashtakar Meh Ahval)

**File:** `src/parsers/column5_parser.py`

This is the MOST CRITICAL component. Column 5 contains semi-structured text like:
```
kasht sahid v singh pisar attar singh kaum sukh sakindeh gair morosi
```

Which needs to be parsed into:
- Name: "sahid v singh"
- Parentage: "attar singh" (son of)
- Caste: "sukh"
- Village: "sakindeh" (same village)
- Remarks: "gair morosi" (non-heirs)

**Key Implementation:**
```python
class Column5Parser:
    """Parser for nam_kashtakar_meh_ahval field"""
    
    # Keyword mappings (transliterated Urdu)
    PARENTAGE_MARKERS = {
        'pisar': 'S/o',      # son of
        'pisaran': 'S/o',    # sons of
        'dukhtar': 'D/o',    # daughter of
        'dukhtaran': 'D/o',  # daughters of
        'zoja': 'W/o',       # wife of
        'byuh': 'Widow/o',   # widow of
    }
    
    ROLE_MARKERS = {
        'kasht': 'cultivator',
        'malik': 'owner',
        'bayaan': 'seller',
        'mushtari': 'buyer',
        'wahib': 'gifter',
        'mohoob': 'gift_receiver',
    }
    
    CASTE_MARKER = 'kaum'
    RESIDENCE_MARKERS = ['sakin', 'sakindeh']
    OWNERSHIP_TYPES = ['morosi', 'gair morosi', 'alati']
    
    def parse(self, text: str) -> CultivatorInfo:
        """Parse Column 5 text into structured components"""
        text = text.lower().strip()
        
        result = CultivatorInfo()
        
        # Extract role (kasht, malik, etc.)
        result.role = self._extract_role(text)
        
        # Extract name (text after role marker, before parentage marker)
        result.name = self._extract_name(text)
        
        # Extract parentage
        result.parentage_type, result.parentage_name = self._extract_parentage(text)
        
        # Extract caste
        result.caste = self._extract_caste(text)
        
        # Extract residence
        result.residence = self._extract_residence(text)
        
        # Extract ownership type
        result.ownership_type = self._extract_ownership_type(text)
        
        # Confidence score based on how many fields were extracted
        result.confidence = self._calculate_confidence(result)
        
        return result
    
    def _extract_name(self, text: str) -> str:
        """Extract name - text between role marker and parentage marker"""
        # Find role marker end position
        role_end = 0
        for marker in self.ROLE_MARKERS:
            if marker in text:
                role_end = text.index(marker) + len(marker)
                break
        
        # Find parentage marker start position
        parentage_start = len(text)
        for marker in self.PARENTAGE_MARKERS:
            if marker in text:
                parentage_start = min(parentage_start, text.index(marker))
        
        name = text[role_end:parentage_start].strip()
        return name
    
    def _extract_parentage(self, text: str) -> Tuple[str, str]:
        """Extract parentage type and name"""
        for marker, type_label in self.PARENTAGE_MARKERS.items():
            if marker in text:
                # Get text after marker until caste marker
                start = text.index(marker) + len(marker)
                end = text.find(self.CASTE_MARKER, start)
                if end == -1:
                    end = len(text)
                parentage_name = text[start:end].strip()
                return type_label, parentage_name
        return None, None
    
    def _extract_caste(self, text: str) -> str:
        """Extract caste - text after 'kaum' marker"""
        if self.CASTE_MARKER in text:
            start = text.index(self.CASTE_MARKER) + len(self.CASTE_MARKER)
            # Find next marker (residence or ownership)
            end = len(text)
            for marker in self.RESIDENCE_MARKERS + self.OWNERSHIP_TYPES:
                if marker in text[start:]:
                    end = min(end, start + text[start:].index(marker))
            return text[start:end].strip()
        return None
```

### 3.2 Khasra Number Splitter

**File:** `src/parsers/khasra_splitter.py`

When a farmer owns multiple parcels, Column 7 shows multiple khasra numbers. Each needs its own row.

**Example Input:**
```
Khasra: 162, 166, 2
Area: 4K 1M, 8K 1M, 12K 2M (kitta = total)
Owner: kasht sandhu singh pisar sain kaum rakwal sakindeh alati
```

**Expected Output:** 3 separate rows, each with same owner info but different khasra/area.

**Key Implementation:**
```python
class KhasraSplitter:
    """Split multi-khasra entries into individual rows"""
    
    def split_entry(self, row: pd.Series) -> List[pd.Series]:
        """Split a single row into multiple rows based on khasra numbers"""
        
        khasra_col = row['number_khasra_v_nam_khet']
        area_col = row['raqba_bakiyad_kisam']
        
        # Parse khasra numbers
        khasras = self._parse_khasra_numbers(khasra_col)
        
        # Parse corresponding areas
        areas = self._parse_areas(area_col)
        
        # Check for 'kitta' (sum row) - this indicates end of a farmer's parcels
        is_sum_row = 'kitta' in str(khasra_col).lower()
        
        if is_sum_row:
            # This is a summary row, handle separately
            return [row]  # Keep as-is or mark as summary
        
        # Create individual rows
        result_rows = []
        for i, (khasra, area) in enumerate(zip(khasras, areas)):
            new_row = row.copy()
            new_row['khasra_hal'] = khasra.get('hal')
            new_row['khasra_sabik'] = khasra.get('sabik')
            new_row['area_kanal'] = area.get('kanal')
            new_row['area_marla'] = area.get('marla')
            new_row['kisam_zamin'] = area.get('kisam')
            new_row['is_split_row'] = True
            new_row['split_index'] = i
            result_rows.append(new_row)
        
        return result_rows
    
    def _parse_khasra_numbers(self, text: str) -> List[dict]:
        """Parse khasra numbers from cell text"""
        # Pattern: hal number followed by optional sabik in format hal/sabik
        # Examples: "156", "362/144", "347/91"
        import re
        
        pattern = r'(\d+)(?:/(\d+))?'
        matches = re.findall(pattern, str(text))
        
        results = []
        for match in matches:
            results.append({
                'hal': match[0],
                'sabik': match[1] if match[1] else None
            })
        return results
    
    def _parse_areas(self, text: str) -> List[dict]:
        """Parse area measurements from cell text"""
        # Pattern: kanal marla format, e.g., "4 1" or "8 10"
        # With land type like "gora nahri", "nahri", "hil nahri"
        import re
        
        # Split by land type keywords
        land_types = ['gora nahri', 'nahri', 'hil nahri', 'gair mumakin', 'bagh nahri']
        
        results = []
        # Implementation depends on exact format in documents
        # ...
        
        return results
```

### 3.3 Row Splitter by Khata

**File:** `src/parsers/jamabandi_parser.py`

Main orchestrator for Jamabandi-specific parsing.

```python
class JamabandiParser:
    """Main parser orchestrating all Jamabandi-specific transformations"""
    
    def __init__(self):
        self.column5_parser = Column5Parser()
        self.khasra_splitter = KhasraSplitter()
    
    def process_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process extracted table into LRIS-compliant format"""
        
        # Step 1: Identify row boundaries by Khata number
        df = self._split_by_khata(df)
        
        # Step 2: Parse Column 5 (Nam Kashtakar)
        df = self._parse_cultivator_info(df)
        
        # Step 3: Split by Khasra numbers
        df = self._expand_khasra_rows(df)
        
        # Step 4: Clean and standardize
        df = self._standardize_columns(df)
        
        return df
    
    def _split_by_khata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split rows based on Khata number changes"""
        # Khata number (Column 2) defines logical row boundaries
        # ...
        pass
    
    def _parse_cultivator_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Column5Parser to all cultivator entries"""
        cultivator_data = []
        for idx, row in df.iterrows():
            col5_text = row.get('nam_kashtakar_meh_ahval', '')
            parsed = self.column5_parser.parse(col5_text)
            cultivator_data.append(parsed.to_dict())
        
        # Add parsed columns to dataframe
        parsed_df = pd.DataFrame(cultivator_data)
        return pd.concat([df, parsed_df], axis=1)
```

---

## 🔨 Phase 4: LLM Integration (2 hours)

### 4.1 LLM Processor for Complex Cells

**File:** `src/processors/llm_processor.py`

For cells that rule-based parsing cannot handle.

```python
class LLMProcessor:
    """Use LLM for complex cell parsing"""
    
    SYSTEM_PROMPT = """You are an expert at parsing Jamabandi (land record) documents from Jammu & Kashmir.
    
The documents use transliterated Urdu with specific terminology:
- kasht = cultivator
- pisar/pisaran = son of/sons of
- dukhtar/dukhtaran = daughter of/daughters of
- zoja = wife of
- byuh = widow of
- kaum = caste
- sakin/sakindeh = resident/resident of same village
- morosi = hereditary
- gair morosi = non-heirs
- alati = temporary

Parse the given text and extract structured information."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic()
    
    def parse_cultivator_field(self, text: str) -> dict:
        """Parse complex Column 5 text using LLM"""
        
        prompt = f"""Parse this Jamabandi cultivator field into structured JSON:

Text: "{text}"

Extract:
1. cultivator_name: The name of the cultivator (after 'kasht')
2. parentage_type: S/o, D/o, W/o, or Widow/o
3. parentage_name: Name of father/husband
4. caste: After 'kaum'
5. residence: After 'sakin' or 'sakindeh'
6. ownership_type: morosi (hereditary), gair morosi (non-heirs), alati (temporary)
7. share_fraction: If ownership share mentioned (e.g., 1/3, 1/10)

Return ONLY valid JSON, no explanation."""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(response.choices[0].message.content)
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"{self.SYSTEM_PROMPT}\n\n{prompt}"}
                ]
            )
            # Parse JSON from response
            return json.loads(response.content[0].text)
    
    def parse_full_row(self, row_text: str) -> dict:
        """Parse entire Jamabandi row when table extraction fails"""
        
        prompt = f"""Parse this Jamabandi land record row into structured JSON:

Raw text: "{row_text}"

Extract all 12 columns:
1. number_khevat: Khevat number
2. number_khata: Khata number
3. nam_tarf_ya_patti: Nambardar name
4. nam_malik_meh_ahval: Owner details
5. nam_kashtakar_meh_ahval: Cultivator details (parse into name, parentage, caste, village)
6. vasayil_abapashi: Irrigation type
7. khasra_number: Survey numbers (hal and sabik)
8. raqba_area: Area in kanal and marla
9. kisam_zamin: Land type
10. lagan_details: Tax details
11. havala_intakal: Mutation reference
12. kaifiyat: Remarks

Return ONLY valid JSON."""

        # Similar implementation as above
        pass
```

### 4.2 Confidence-Based Routing

```python
class HybridProcessor:
    """Route between rule-based and LLM processing based on confidence"""
    
    def __init__(self, llm_threshold: float = 0.7):
        self.column5_parser = Column5Parser()
        self.llm_processor = LLMProcessor()
        self.llm_threshold = llm_threshold
    
    def process_cultivator_field(self, text: str) -> dict:
        """Process with rule-based first, fallback to LLM if low confidence"""
        
        # Try rule-based first
        result = self.column5_parser.parse(text)
        
        if result.confidence >= self.llm_threshold:
            return result.to_dict()
        
        # Fallback to LLM for low-confidence cases
        llm_result = self.llm_processor.parse_cultivator_field(text)
        llm_result['extraction_method'] = 'llm'
        
        return llm_result
```

---

## 🔨 Phase 5: Validation & Output (2 hours)

### 5.1 Pydantic Data Models

**File:** `src/validators/models.py`

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum

class ParentageType(str, Enum):
    SON_OF = "S/o"
    DAUGHTER_OF = "D/o"
    WIFE_OF = "W/o"
    WIDOW_OF = "Widow/o"

class OwnershipType(str, Enum):
    HEREDITARY = "morosi"
    NON_HEIRS = "gair_morosi"
    TEMPORARY = "alati"

class CultivatorInfo(BaseModel):
    """Parsed cultivator information from Column 5"""
    name: str
    parentage_type: Optional[ParentageType]
    parentage_name: Optional[str]
    caste: Optional[str]
    residence: Optional[str]
    ownership_type: Optional[OwnershipType]
    share_fraction: Optional[str]
    role: str = "cultivator"
    confidence: float = Field(ge=0, le=1)
    
    @validator('name')
    def name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Cultivator name cannot be empty')
        return v.strip()

class KhasraInfo(BaseModel):
    """Khasra (survey) number information"""
    hal: str  # Current survey number
    sabik: Optional[str]  # Previous survey number

class AreaInfo(BaseModel):
    """Land area information"""
    kanal: int = Field(ge=0)
    marla: int = Field(ge=0, le=20)
    kisam_zamin: Optional[str]  # Land type

class JamabandiRecord(BaseModel):
    """Complete Jamabandi record after parsing"""
    # Identifiers
    number_khevat: int
    number_khata: int
    
    # Names
    nam_tarf_ya_patti: Optional[str]
    nam_malik_meh_ahval: Optional[str]
    
    # Cultivator (parsed Column 5)
    cultivator: CultivatorInfo
    
    # Land details
    vasayil_abapashi: Optional[str]  # Irrigation
    khasra: KhasraInfo
    area: AreaInfo
    
    # Transaction details
    lagan_details: Optional[str]
    mutalba_details: Optional[str]
    havala_intakal: Optional[str]  # Mutation reference
    kaifiyat: Optional[str]  # Remarks
    
    # Metadata
    extraction_confidence: float
    extraction_method: str
    page_number: int
    is_sum_row: bool = False

class VillageJamabandi(BaseModel):
    """Complete Jamabandi for a village"""
    # Header info
    sal: str  # Year
    zla: str  # District
    thsil: str  # Tehsil
    babata_mouza: str  # Village
    
    # Records
    records: List[JamabandiRecord]
    
    # Processing metadata
    total_pages: int
    total_records: int
    average_confidence: float
    low_confidence_count: int
```

### 5.2 Excel Exporter

**File:** `src/utils/excel_exporter.py`

```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

class LRISExcelExporter:
    """Export parsed data to LRIS-compliant Excel format"""
    
    COLUMNS = [
        'number_khevat', 'number_khata', 'nam_tarf_ya_patti',
        'nam_malik_meh_ahval', 'cultivator_name', 'cultivator_parentage',
        'cultivator_caste', 'cultivator_village', 'cultivator_remarks',
        'vasayil_abapashi', 'khasra_hal', 'khasra_sabik',
        'area_kanal', 'area_marla', 'kisam_zamin',
        'lagan_details', 'mutalba_details', 'havala_intakal', 'kaifiyat',
        'extraction_confidence', 'needs_review'
    ]
    
    def export(self, records: List[JamabandiRecord], output_path: str):
        """Export records to Excel"""
        
        # Convert to flat DataFrame
        data = []
        for record in records:
            row = {
                'number_khevat': record.number_khevat,
                'number_khata': record.number_khata,
                'nam_tarf_ya_patti': record.nam_tarf_ya_patti,
                'nam_malik_meh_ahval': record.nam_malik_meh_ahval,
                'cultivator_name': record.cultivator.name,
                'cultivator_parentage': f"{record.cultivator.parentage_type} {record.cultivator.parentage_name}" if record.cultivator.parentage_type else "",
                'cultivator_caste': record.cultivator.caste,
                'cultivator_village': record.cultivator.residence,
                'cultivator_remarks': record.cultivator.ownership_type,
                'vasayil_abapashi': record.vasayil_abapashi,
                'khasra_hal': record.khasra.hal,
                'khasra_sabik': record.khasra.sabik,
                'area_kanal': record.area.kanal,
                'area_marla': record.area.marla,
                'kisam_zamin': record.area.kisam_zamin,
                'lagan_details': record.lagan_details,
                'mutalba_details': record.mutalba_details,
                'havala_intakal': record.havala_intakal,
                'kaifiyat': record.kaifiyat,
                'extraction_confidence': record.extraction_confidence,
                'needs_review': record.extraction_confidence < 0.8
            }
            data.append(row)
        
        df = pd.DataFrame(data, columns=self.COLUMNS)
        
        # Export with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Land Records', index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Land Records']
            
            # Apply formatting
            self._apply_formatting(worksheet)
        
        return output_path
    
    def _apply_formatting(self, worksheet):
        """Apply Excel formatting for readability"""
        # Header formatting
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')
        
        # Highlight low confidence rows
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        
        for row in worksheet.iter_rows(min_row=2):
            # Check needs_review column (last column)
            if row[-1].value == True:
                for cell in row:
                    cell.fill = yellow_fill
```

---

## 🔨 Phase 6: Main Pipeline & CLI (1 hour)

### 6.1 Main Entry Point

**File:** `src/main.py`

```python
import argparse
from pathlib import Path
from typing import Optional
import logging

from extractors.pdf_loader import JamabandiPDFLoader
from extractors.camelot_extractor import CamelotExtractor
from extractors.pdfplumber_extractor import PdfplumberExtractor
from parsers.jamabandi_parser import JamabandiParser
from processors.llm_processor import HybridProcessor
from validators.models import VillageJamabandi
from utils.excel_exporter import LRISExcelExporter
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

class JamabandiPipeline:
    """Main pipeline for Jamabandi extraction and processing"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.pdf_loader = JamabandiPDFLoader
        self.camelot_extractor = CamelotExtractor(
            accuracy_threshold=self.config.get('camelot_accuracy_threshold', 75)
        )
        self.pdfplumber_extractor = PdfplumberExtractor()
        self.parser = JamabandiParser()
        self.hybrid_processor = HybridProcessor(
            llm_threshold=self.config.get('llm_threshold', 0.7)
        )
        self.exporter = LRISExcelExporter()
    
    def process(self, pdf_path: str, output_path: str) -> VillageJamabandi:
        """Process a single Jamabandi PDF"""
        
        logger.info(f"Processing: {pdf_path}")
        
        # Step 1: Load PDF and extract metadata
        loader = self.pdf_loader(pdf_path)
        metadata = loader.metadata
        logger.info(f"Village: {metadata['babata_mouza']}, District: {metadata['zla']}")
        
        # Step 2: Extract tables
        tables = self.camelot_extractor.extract_tables(pdf_path)
        
        # Check accuracy and use fallback if needed
        low_accuracy_pages = []
        for table in tables:
            if table.accuracy < self.config.get('camelot_accuracy_threshold', 75):
                low_accuracy_pages.append(table.page)
                logger.warning(f"Low accuracy on page {table.page}: {table.accuracy}%")
        
        # Use pdfplumber for low-accuracy pages
        if low_accuracy_pages:
            fallback_tables = self.pdfplumber_extractor.extract_tables(
                pdf_path, 
                pages=low_accuracy_pages
            )
            # Merge results
            # ...
        
        # Step 3: Parse Jamabandi-specific structure
        all_records = []
        for table in tables:
            parsed_df = self.parser.process_table(table.dataframe)
            
            # Step 4: Process complex cells with hybrid approach
            for idx, row in parsed_df.iterrows():
                cultivator_text = row.get('nam_kashtakar_meh_ahval', '')
                parsed_cultivator = self.hybrid_processor.process_cultivator_field(cultivator_text)
                # Update row with parsed data
                # ...
            
            all_records.extend(parsed_df.to_dict('records'))
        
        # Step 5: Validate and create output
        village_data = VillageJamabandi(
            sal=metadata['sal'],
            zla=metadata['zla'],
            thsil=metadata['thsil'],
            babata_mouza=metadata['babata_mouza'],
            records=all_records,
            total_pages=len(loader.doc),
            total_records=len(all_records),
            average_confidence=sum(r['extraction_confidence'] for r in all_records) / len(all_records),
            low_confidence_count=sum(1 for r in all_records if r['extraction_confidence'] < 0.8)
        )
        
        # Step 6: Export to Excel
        self.exporter.export(village_data.records, output_path)
        
        logger.info(f"Exported {len(all_records)} records to {output_path}")
        logger.info(f"Average confidence: {village_data.average_confidence:.2%}")
        logger.info(f"Records needing review: {village_data.low_confidence_count}")
        
        return village_data


def main():
    parser = argparse.ArgumentParser(description='J&K Jamabandi Land Record Extractor')
    parser.add_argument('--input', '-i', required=True, help='Input PDF path')
    parser.add_argument('--output', '-o', required=True, help='Output Excel path')
    parser.add_argument('--config', '-c', help='Config file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose)
    
    pipeline = JamabandiPipeline(config_path=args.config)
    result = pipeline.process(args.input, args.output)
    
    print(f"\n✅ Processing complete!")
    print(f"   Records extracted: {result.total_records}")
    print(f"   Average confidence: {result.average_confidence:.2%}")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
```

---

## 📋 Testing Checklist

### Unit Tests
- [ ] Column5Parser correctly extracts all field types
- [ ] KhasraSplitter handles single and multiple khasra numbers
- [ ] Pydantic models validate correctly
- [ ] Excel export produces valid LRIS format

### Integration Tests
- [ ] Full pipeline processes sample PDF without errors
- [ ] Output Excel matches expected schema
- [ ] LLM fallback triggers for low-confidence cases

### Validation Tests
- [ ] Manual comparison with ground truth for 2-3 pages
- [ ] Accuracy metrics calculated and logged
- [ ] Edge cases handled (empty cells, malformed data)

---

## 📊 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Extraction Accuracy | >85% | Compare with manual ground truth |
| Column 5 Parse Rate | >80% | % of records with all fields extracted |
| Processing Speed | >1 page/sec | Time logged per page |
| Error Rate | <5% | Records requiring manual correction |
| Schema Compliance | 100% | Pydantic validation pass rate |

---

## 🚀 Deployment for Demo

For the hackathon demo, create a simple Streamlit app:

```python
# src/app.py
import streamlit as st
from main import JamabandiPipeline

st.title("🏛️ J&K Land Record Digitization System")
st.subheader("AgriStack - Jamabandi PDF to Excel Converter")

uploaded_file = st.file_uploader("Upload Jamabandi PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing..."):
        pipeline = JamabandiPipeline()
        result = pipeline.process(uploaded_file, "output.xlsx")
    
    st.success(f"✅ Extracted {result.total_records} records")
    st.metric("Confidence", f"{result.average_confidence:.2%}")
    
    # Download button
    with open("output.xlsx", "rb") as f:
        st.download_button("Download Excel", f, "jamabandi_output.xlsx")
```

Run with: `streamlit run src/app.py`