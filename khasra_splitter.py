"""
Khasra Number Splitter

Handles the expansion of rows where a single cultivator owns multiple
land parcels (represented by multiple Khasra numbers in Column 7).

Each Khasra number needs its own row in the output, with the cultivator
information duplicated across all rows.

Example Input (single row):
    Khasra: 162, 166, 2 (kitta)
    Area: 4K 1M, 8K 1M, 12K 2M
    Owner: kasht sandhu singh pisar sain kaum rakwal sakindeh alati

Expected Output (3 rows):
    Row 1: Khasra 162, Area 4K 1M, Owner: sandhu singh...
    Row 2: Khasra 166, Area 8K 1M, Owner: sandhu singh...
    Row 3: Khasra 2 (skip - this is the sum row marked with 'kitta')
"""

import re
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KhasraEntry:
    """Single Khasra (survey number) entry"""
    hal: str  # Current survey number
    sabik: Optional[str] = None  # Previous survey number (if in hal/sabik format)
    
    def __str__(self) -> str:
        if self.sabik:
            return f"{self.hal}/{self.sabik}"
        return self.hal


@dataclass
class AreaEntry:
    """Single area measurement entry"""
    kanal: int
    marla: int
    kisam_zamin: Optional[str] = None  # Land type
    
    @property
    def total_marla(self) -> int:
        """Convert to total marla (20 marla = 1 kanal)"""
        return (self.kanal * 20) + self.marla
    
    def __str__(self) -> str:
        type_str = f" ({self.kisam_zamin})" if self.kisam_zamin else ""
        return f"{self.kanal}K {self.marla}M{type_str}"


class KhasraSplitter:
    """
    Splits multi-Khasra rows into individual rows.
    
    Handles:
    - Multiple Khasra numbers in a single cell
    - Corresponding area entries
    - Sum rows marked with 'kitta'
    - Format variations in Khasra numbers (hal or hal/sabik)
    """
    
    # Pattern for Khasra numbers: hal or hal/sabik
    KHASRA_PATTERN = re.compile(r'(\d+)(?:/(\d+))?')
    
    # Pattern for area: kanal marla
    AREA_PATTERN = re.compile(r'(\d+)\s+(\d+)')
    
    # Sum row marker
    SUM_MARKER = 'kitta'
    
    # Known land types (kisam zamin)
    LAND_TYPES = [
        'gora nahri',     # Unirrigated near canal
        'nahri',          # Canal irrigated
        'hil nahri',      # Hill irrigated
        'bagh nahri',     # Garden irrigated
        'gair mumakin',   # Uncultivable
        'gair mumkin abadi',  # Uncultivable settlement
        'gair mumakin kuhl',  # Uncultivable channel
    ]
    
    def __init__(self, preserve_sum_rows: bool = False):
        """
        Initialize splitter.
        
        Args:
            preserve_sum_rows: If True, keep 'kitta' rows marked. If False, drop them.
        """
        self.preserve_sum_rows = preserve_sum_rows
    
    def is_sum_row(self, row: pd.Series) -> bool:
        """Check if row is a sum/total row marked with 'kitta'"""
        # Check common columns where 'kitta' might appear
        for col in row.index:
            val = str(row[col]).lower()
            if self.SUM_MARKER in val:
                return True
        return False
    
    def split_row(self, row: pd.Series, khasra_col: str, area_col: str) -> List[pd.Series]:
        """
        Split a single row into multiple rows based on Khasra numbers.
        
        Args:
            row: DataFrame row to split
            khasra_col: Column name containing Khasra numbers
            area_col: Column name containing area values
            
        Returns:
            List of rows (original or split)
        """
        # Check for sum row
        if self.is_sum_row(row):
            logger.debug(f"Sum row detected, skipping split")
            if self.preserve_sum_rows:
                new_row = row.copy()
                new_row['is_sum_row'] = True
                return [new_row]
            return []  # Drop sum rows
        
        # Extract Khasra numbers
        khasra_text = str(row.get(khasra_col, ''))
        khasras = self.parse_khasra_numbers(khasra_text)
        
        if not khasras:
            logger.warning(f"No Khasra numbers found in: {khasra_text}")
            return [row]  # Return original row
        
        # If only one Khasra, no split needed
        if len(khasras) == 1:
            new_row = row.copy()
            new_row['khasra_hal'] = khasras[0].hal
            new_row['khasra_sabik'] = khasras[0].sabik
            new_row['is_split_row'] = False
            return [new_row]
        
        # Extract areas
        area_text = str(row.get(area_col, ''))
        areas = self.parse_areas(area_text)
        
        # Create split rows
        result_rows = []
        for i, khasra in enumerate(khasras):
            new_row = row.copy()
            
            # Set Khasra values
            new_row['khasra_hal'] = khasra.hal
            new_row['khasra_sabik'] = khasra.sabik
            
            # Set area values (if available)
            if i < len(areas):
                new_row['area_kanal'] = areas[i].kanal
                new_row['area_marla'] = areas[i].marla
                new_row['kisam_zamin'] = areas[i].kisam_zamin
            
            # Mark as split row
            new_row['is_split_row'] = True
            new_row['split_index'] = i
            new_row['split_total'] = len(khasras)
            
            result_rows.append(new_row)
        
        logger.debug(f"Split into {len(result_rows)} rows from {len(khasras)} Khasras")
        return result_rows
    
    def parse_khasra_numbers(self, text: str) -> List[KhasraEntry]:
        """
        Parse Khasra numbers from cell text.
        
        Handles formats:
        - Single: "156"
        - With sabik: "362/144"
        - Multiple: "162 166 2" or "162, 166, 2"
        - Mixed: "362/144 386/329"
        """
        results = []
        
        # Remove 'kitta' and other markers
        cleaned = text.lower().replace(self.SUM_MARKER, '').strip()
        
        # Remove land type words to avoid false matches
        for land_type in self.LAND_TYPES:
            cleaned = cleaned.replace(land_type, ' ')
        
        # Find all Khasra patterns
        matches = self.KHASRA_PATTERN.findall(cleaned)
        
        for match in matches:
            hal = match[0]
            sabik = match[1] if match[1] else None
            
            # Skip very small numbers that might be area values
            # Khasra numbers are typically 2+ digits
            if len(hal) < 2 and not sabik:
                continue
            
            results.append(KhasraEntry(hal=hal, sabik=sabik))
        
        return results
    
    def parse_areas(self, text: str) -> List[AreaEntry]:
        """
        Parse area measurements from cell text.
        
        Format: kanal marla, e.g., "4 1" means 4 kanal 1 marla
        
        Also extracts land type (kisam zamin) if present.
        """
        results = []
        
        # Split by land types to get individual area entries
        remaining_text = text.lower()
        
        # First, identify land types and their positions
        land_type_positions = []
        for land_type in sorted(self.LAND_TYPES, key=len, reverse=True):
            for match in re.finditer(re.escape(land_type), remaining_text):
                land_type_positions.append((match.start(), match.end(), land_type))
        
        # Sort by position
        land_type_positions.sort(key=lambda x: x[0])
        
        # Extract area numbers with their associated land types
        area_matches = list(self.AREA_PATTERN.finditer(remaining_text))
        
        for i, match in enumerate(area_matches):
            kanal = int(match.group(1))
            marla = int(match.group(2))
            
            # Find associated land type (nearest one before or after)
            kisam = None
            match_pos = match.start()
            
            for start, end, land_type in land_type_positions:
                # Land type should be near the area values
                if abs(start - match_pos) < 30 or abs(end - match_pos) < 30:
                    kisam = land_type
                    break
            
            results.append(AreaEntry(kanal=kanal, marla=marla, kisam_zamin=kisam))
        
        return results
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        khasra_col: str = 'number_khasra_v_nam_khet',
        area_col: str = 'raqba_bakiyad_kisam'
    ) -> pd.DataFrame:
        """
        Process entire DataFrame, splitting multi-Khasra rows.
        
        Args:
            df: Input DataFrame
            khasra_col: Column containing Khasra numbers
            area_col: Column containing area values
            
        Returns:
            Expanded DataFrame with one row per Khasra
        """
        logger.info(f"Processing {len(df)} rows for Khasra splitting")
        
        all_rows = []
        
        for idx, row in df.iterrows():
            split_rows = self.split_row(row, khasra_col, area_col)
            all_rows.extend(split_rows)
        
        result_df = pd.DataFrame(all_rows)
        
        logger.info(f"Expanded to {len(result_df)} rows")
        
        # Statistics
        sum_rows = result_df['is_sum_row'].sum() if 'is_sum_row' in result_df.columns else 0
        split_rows = result_df['is_split_row'].sum() if 'is_split_row' in result_df.columns else 0
        
        logger.info(f"  Sum rows: {sum_rows}")
        logger.info(f"  Split rows: {split_rows}")
        
        return result_df


class KhasraAreaMatcher:
    """
    Advanced matching of Khasra numbers to their corresponding areas.
    
    Handles complex cases where the PDF layout makes direct position
    mapping unreliable.
    """
    
    def __init__(self):
        self.splitter = KhasraSplitter()
    
    def match_khasra_to_area(
        self,
        khasra_text: str,
        area_text: str
    ) -> List[Tuple[KhasraEntry, Optional[AreaEntry]]]:
        """
        Match Khasra numbers to their areas.
        
        Uses position-based and count-based heuristics.
        """
        khasras = self.splitter.parse_khasra_numbers(khasra_text)
        areas = self.splitter.parse_areas(area_text)
        
        results = []
        
        # If counts match, use direct pairing
        if len(khasras) == len(areas):
            for k, a in zip(khasras, areas):
                results.append((k, a))
        
        # If more areas than Khasras, last might be sum
        elif len(areas) > len(khasras):
            for i, k in enumerate(khasras):
                results.append((k, areas[i] if i < len(areas) else None))
        
        # If more Khasras than areas, some areas are missing
        else:
            for i, k in enumerate(khasras):
                results.append((k, areas[i] if i < len(areas) else None))
        
        return results


# === Testing ===

if __name__ == "__main__":
    # Test cases based on actual Jamabandi data
    
    test_cases = [
        # Single Khasra
        {
            'khasra': '156',
            'area': '16 0 gora nahri',
            'expected_count': 1
        },
        # Multiple Khasras
        {
            'khasra': '162 166 2 kitta',
            'area': '4 1 gora nahri 8 1 nahri 12 2',
            'expected_count': 2  # kitta row should be dropped
        },
        # With sabik format
        {
            'khasra': '362/144 386/329',
            'area': '0 1 hil nahri 8 4 nahri',
            'expected_count': 2
        },
        # Complex multi-khasra
        {
            'khasra': '76 69 69 69 69 216',
            'area': '8 10 nahri 18 7 nahri 5 3 nahri 10 0 nahri 9 1 nahri',
            'expected_count': 6
        }
    ]
    
    splitter = KhasraSplitter()
    
    print("=" * 70)
    print("Khasra Splitter Test Results")
    print("=" * 70)
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Khasra text: {test['khasra']}")
        print(f"  Area text: {test['area']}")
        
        khasras = splitter.parse_khasra_numbers(test['khasra'])
        areas = splitter.parse_areas(test['area'])
        
        print(f"  Parsed Khasras: {[str(k) for k in khasras]}")
        print(f"  Parsed Areas: {[str(a) for a in areas]}")
        print(f"  Expected count: {test['expected_count']}")
        print(f"  Actual count: {len(khasras)}")
        
        # Simulate row splitting
        row = pd.Series({
            'number_khasra_v_nam_khet': test['khasra'],
            'raqba_bakiyad_kisam': test['area'],
            'cultivator': 'test cultivator'
        })
        
        split_rows = splitter.split_row(
            row,
            khasra_col='number_khasra_v_nam_khet',
            area_col='raqba_bakiyad_kisam'
        )
        
        print(f"  Split into {len(split_rows)} rows")