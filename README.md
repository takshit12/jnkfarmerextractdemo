"""
Main Jamabandi Extraction Pipeline

Orchestrates the complete workflow:
1. PDF Loading → 2. Table Extraction → 3. Jamabandi Parsing → 
4. LLM Processing → 5. Validation → 6. Excel Export

Usage:
    python -m src.main --input data/input/village.pdf --output data/output/
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import json

import pandas as pd
import yaml
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors.camelot_extractor import CamelotExtractor, ExtractedTable
from parsers.column5_parser import Column5Parser
from parsers.khasra_splitter import KhasraSplitter
from validators.models import (
    JamabandiRecord, VillageJamabandi, VillageMetadata,
    CultivatorInfo, KhasraInfo, AreaInfo, MutationInfo,
    ExtractionMethod, create_empty_cultivator
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class JamabandiPipeline:
    """
    Main pipeline for Jamabandi land record extraction.
    
    Orchestrates all components to produce structured Excel output.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to settings.yaml (optional)
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.camelot_extractor = CamelotExtractor(
            accuracy_threshold=self.config.get('extraction', {}).get('camelot', {}).get('accuracy_threshold', 75),
            line_scale=self.config.get('extraction', {}).get('camelot', {}).get('line_scale', 40)
        )
        
        self.column5_parser = Column5Parser()
        self.khasra_splitter = KhasraSplitter(preserve_sum_rows=False)
        
        # LLM processor (lazy load)
        self._llm_processor = None
        
        logger.info("Jamabandi Pipeline initialized")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Try default location
        default_path = Path(__file__).parent.parent.parent / 'config' / 'settings.yaml'
        if default_path.exists():
            with open(default_path, 'r') as f:
                return yaml.safe_load(f)
        
        logger.warning("No config file found, using defaults")
        return {}
    
    @property
    def llm_processor(self):
        """Lazy load LLM processor"""
        if self._llm_processor is None:
            try:
                from processors.llm_processor import LLMProcessor
                self._llm_processor = LLMProcessor(
                    provider=self.config.get('llm', {}).get('provider', 'openai'),
                    model=self.config.get('llm', {}).get('openai', {}).get('model', 'gpt-4o')
                )
            except ImportError:
                logger.warning("LLM processor not available")
                self._llm_processor = None
        return self._llm_processor
    
    def process(
        self,
        pdf_path: str,
        output_path: str,
        pages: str = "all"
    ) -> VillageJamabandi:
        """
        Process a Jamabandi PDF and generate structured output.
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output Excel file
            pages: Pages to process ("all", "1", "1-5", "1,2,3")
            
        Returns:
            VillageJamabandi with all extracted records
        """
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)
        
        logger.info(f"Processing: {pdf_path}")
        start_time = datetime.now()
        
        # === Step 1: Extract Metadata ===
        metadata = self._extract_metadata(pdf_path)
        logger.info(f"Village: {metadata.babata_mouza}, District: {metadata.zla}")
        
        # === Step 2: Extract Tables ===
        logger.info("Extracting tables with Camelot...")
        tables, fallback_pages = self.camelot_extractor.extract_with_fallback_info(
            str(pdf_path), pages=pages
        )
        
        if not tables:
            raise ValueError("No tables extracted from PDF")
        
        logger.info(f"Extracted {len(tables)} tables")
        
        # === Step 3: Process Each Table ===
        all_records = []
        
        for table in tables:
            logger.info(f"Processing page {table.page} (accuracy: {table.accuracy:.1f}%)")
            
            # Parse table into records
            records = self._process_table(table, table.page)
            all_records.extend(records)
        
        # === Step 4: Create Output ===
        village_data = VillageJamabandi(
            metadata=metadata,
            records=all_records,
            source_file=pdf_path.name,
            total_pages=len(tables)
        )
        
        # === Step 5: Export to Excel ===
        self._export_to_excel(village_data, output_path)
        
        # === Log Summary ===
        duration = (datetime.now() - start_time).total_seconds()
        summary = village_data.get_summary()
        
        logger.info("=" * 50)
        logger.info("Processing Complete")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Records: {summary['total_records']}")
        logger.info(f"  Avg Confidence: {summary['average_confidence']:.1%}")
        logger.info(f"  Needs Review: {summary['records_needing_review']} ({summary['review_percentage']:.1f}%)")
        logger.info(f"  Output: {output_path}")
        logger.info("=" * 50)
        
        return village_data
    
    def _extract_metadata(self, pdf_path: Path) -> VillageMetadata:
        """
        Extract village metadata from PDF header.
        
        Header format: "2017-18 : sal ... Jammu :zla ... :thsil ... :babata mouza"
        """
        import fitz  # PyMuPDF
        
        doc = fitz.open(str(pdf_path))
        first_page_text = doc[0].get_text()
        doc.close()
        
        # Parse header
        # Pattern: "2017-18 : sal ... Jammu :zla ... Jammu West :thsil ... Gujral :babata mouza"
        
        metadata = {
            'sal': '2017-18',  # Default
            'zla': 'Unknown',
            'thsil': 'Unknown',
            'babata_mouza': pdf_path.stem  # Use filename as fallback
        }
        
        # Try to extract from text
        lines = first_page_text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line_lower = line.lower()
            
            if ':sal' in line_lower or ': sal' in line_lower:
                # Year
                import re
                year_match = re.search(r'(\d{4}-\d{2})', line)
                if year_match:
                    metadata['sal'] = year_match.group(1)
            
            if ':zla' in line_lower:
                # District - text before :zla
                parts = line.split(':zla')
                if len(parts) > 0:
                    district = parts[0].strip().split()[-1] if parts[0].strip() else 'Unknown'
                    metadata['zla'] = district
            
            if ':thsil' in line_lower:
                # Tehsil - text before :thsil
                parts = line.split(':thsil')
                if len(parts) > 0:
                    tehsil = parts[0].strip().split(':')[-1].strip() if parts[0].strip() else 'Unknown'
                    metadata['thsil'] = tehsil
            
            if ':babata mouza' in line_lower or ':babata' in line_lower:
                # Village - text before :babata
                parts = line.lower().split(':babata')
                if len(parts) > 0:
                    village = parts[0].strip().split(':')[-1].strip() if parts[0].strip() else pdf_path.stem
                    metadata['babata_mouza'] = village.title()
        
        return VillageMetadata(**metadata)
    
    def _process_table(
        self,
        table: ExtractedTable,
        page_number: int
    ) -> List[JamabandiRecord]:
        """
        Process extracted table into Jamabandi records.
        
        Steps:
        1. Map columns to schema
        2. Parse Column 5 (cultivator info)
        3. Split by Khasra numbers
        4. Create validated records
        """
        df = table.dataframe
        records = []
        
        # Try to identify columns
        column_map = self._identify_columns(df)
        
        for row_idx, row in df.iterrows():
            try:
                # Skip header rows
                if self._is_header_row(row):
                    continue
                
                # Extract basic fields
                khevat = self._safe_int(row.iloc[column_map.get('khevat', 0)])
                khata = self._safe_int(row.iloc[column_map.get('khata', 1)])
                
                # Skip if no khata
                if khata == 0:
                    continue
                
                # Parse cultivator info (Column 5)
                cultivator_text = str(row.iloc[column_map.get('cultivator', 4)])
                cultivator_info = self.column5_parser.parse(cultivator_text)
                
                # Check if LLM fallback needed
                if cultivator_info.confidence < 0.7 and self.llm_processor:
                    try:
                        llm_result = self.llm_processor.parse_cultivator_field(cultivator_text)
                        if llm_result:
                            cultivator_info = CultivatorInfo(
                                **llm_result,
                                raw_text=cultivator_text,
                                confidence=0.9,
                                extraction_method=ExtractionMethod.LLM
                            )
                    except Exception as e:
                        logger.warning(f"LLM processing failed: {e}")
                
                # Parse Khasra and Area
                khasra_text = str(row.iloc[column_map.get('khasra', 6)])
                area_text = str(row.iloc[column_map.get('area', 7)])
                
                # Split by Khasra if multiple
                khasras = self.khasra_splitter.parse_khasra_numbers(khasra_text)
                areas = self.khasra_splitter.parse_areas(area_text)
                
                # Skip sum rows
                if self.khasra_splitter.is_sum_row(row):
                    continue
                
                # Create records (one per Khasra)
                for i, khasra in enumerate(khasras):
                    area = areas[i] if i < len(areas) else None
                    
                    record = JamabandiRecord(
                        number_khevat=khevat,
                        number_khata=khata,
                        nam_tarf_ya_patti=str(row.iloc[column_map.get('nambardar', 2)]) if column_map.get('nambardar') else None,
                        nam_malik_meh_ahval=str(row.iloc[column_map.get('malik', 3)]) if column_map.get('malik') else None,
                        cultivator=cultivator_info,
                        vasayil_abapashi=str(row.iloc[column_map.get('irrigation', 5)]) if column_map.get('irrigation') else None,
                        khasra=KhasraInfo(hal=khasra.hal, sabik=khasra.sabik),
                        area=AreaInfo(
                            kanal=area.kanal if area else 0,
                            marla=area.marla if area else 0,
                            kisam_zamin=area.kisam_zamin if area else None
                        ),
                        mutation=self._parse_mutation(row, column_map),
                        page_number=page_number,
                        row_index=row_idx,
                        is_split_row=len(khasras) > 1,
                        split_index=i if len(khasras) > 1 else None,
                        extraction_confidence=cultivator_info.confidence * (table.accuracy / 100),
                        extraction_method=cultivator_info.extraction_method
                    )
                    
                    records.append(record)
                    
            except Exception as e:
                logger.error(f"Error processing row {row_idx}: {e}")
                continue
        
        return records
    
    def _identify_columns(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Identify column positions in the DataFrame.
        
        Jamabandi has 12 columns (right to left):
        1-khevat, 2-khata, 3-nambardar, 4-malik, 5-cultivator,
        6-irrigation, 7-khasra, 8-area, 9-tax, 10-cess, 11-mutation, 12-remarks
        """
        # Default mapping for 12-column Jamabandi
        # Columns in DataFrame are typically left-to-right after extraction
        num_cols = len(df.columns)
        
        if num_cols >= 12:
            return {
                'remarks': 0,      # Column 12 (kaifiyat)
                'mutation': 1,     # Column 11 (havala intakal)
                'cess': 2,         # Column 10
                'tax': 3,          # Column 9
                'area': 4,         # Column 8
                'khasra': 5,       # Column 7
                'irrigation': 6,   # Column 6
                'cultivator': 7,   # Column 5
                'malik': 8,        # Column 4
                'nambardar': 9,    # Column 3
                'khata': 10,       # Column 2
                'khevat': 11,      # Column 1
            }
        else:
            # Simplified mapping for fewer columns
            logger.warning(f"Only {num_cols} columns found, using simplified mapping")
            return {
                'khevat': num_cols - 1,
                'khata': num_cols - 2 if num_cols > 1 else 0,
                'cultivator': min(4, num_cols - 1),
                'khasra': min(5, num_cols - 1),
                'area': min(6, num_cols - 1),
            }
    
    def _is_header_row(self, row: pd.Series) -> bool:
        """Check if row is a header row"""
        header_keywords = [
            'number khata', 'number khevat', 'kaifiyat', 'nam malik',
            'nam kashtakar', 'khasra', 'raqba', 'vasayil'
        ]
        
        row_text = ' '.join(str(v).lower() for v in row.values)
        
        for keyword in header_keywords:
            if keyword in row_text:
                return True
        return False
    
    def _parse_mutation(self, row: pd.Series, column_map: Dict[str, int]) -> Optional[MutationInfo]:
        """Parse mutation information from row"""
        if 'mutation' not in column_map:
            return None
        
        mutation_text = str(row.iloc[column_map['mutation']])
        
        if not mutation_text or mutation_text.lower() in ['nan', 'none', '']:
            return None
        
        # Try to extract type and number
        import re
        
        # Common mutation types
        mutation_types = ['hembba nama', 'varasat', 'wasit morosi', 'tht dfah']
        
        kisam = None
        for mtype in mutation_types:
            if mtype in mutation_text.lower():
                kisam = mtype
                break
        
        # Extract number
        number_match = re.search(r'\d+', mutation_text)
        number = number_match.group() if number_match else None
        
        return MutationInfo(kisam=kisam, number=number, raw_text=mutation_text)
    
    def _safe_int(self, value) -> int:
        """Safely convert value to int"""
        try:
            if pd.isna(value):
                return 0
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return 0
    
    def _export_to_excel(self, village_data: VillageJamabandi, output_path: Path) -> None:
        """Export records to Excel file"""
        # Convert records to flat dictionaries
        data = [record.to_flat_dict() for record in village_data.records]
        
        if not data:
            logger.warning("No data to export")
            return
        
        df = pd.DataFrame(data)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Land Records', index=False)
            
            # Add summary sheet
            summary_data = [village_data.get_summary()]
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Exported to: {output_path}")


# === CLI Interface ===

def main():
    """Command-line interface for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='J&K Jamabandi Land Record Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single PDF
    python -m src.main --input data/input/gujral.pdf --output data/output/gujral.xlsx
    
    # Process specific pages
    python -m src.main --input data/input/gujral.pdf --output data/output/ --pages 1-5
    
    # Verbose mode
    python -m src.main --input data/input/gujral.pdf --output data/output/ -v
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input PDF path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output Excel path or directory'
    )
    
    parser.add_argument(
        '--pages', '-p',
        default='all',
        help='Pages to process (default: all). Examples: "1", "1-5", "1,3,5"'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('extraction.log')
        ]
    )
    
    # Determine output path
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if output_path.is_dir():
        output_path = output_path / f"{input_path.stem}_output.xlsx"
    
    # Run pipeline
    try:
        pipeline = JamabandiPipeline(config_path=args.config)
        result = pipeline.process(
            pdf_path=str(input_path),
            output_path=str(output_path),
            pages=args.pages
        )
        
        print(f"\n✅ Processing complete!")
        print(f"   Records: {result.total_records}")
        print(f"   Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()