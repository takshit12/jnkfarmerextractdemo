"""
Main Pipeline for Jamabandi Extraction

Integrates all components into a complete, production-ready pipeline.
"""

import sys
from pathlib import Path
from typing import Optional
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import get_logger, get_config, log_context, PerformanceLogger
from src.core.exceptions import AgriStackError
from src.llm import LLMProcessor, LLMRouter
from src.exporters import ExcelExporter

# Import existing components
from camelot_extractor import CamelotExtractor
from column5_parser import Column5Parser
from khasra_splitter import KhasraSplitter
from models import (
    JamabandiRecord, VillageJamabandi, VillageMetadata,
    CultivatorInfo, KhasraInfo, AreaInfo, ExtractionMethod, MutationInfo
)

logger = get_logger(__name__)


class JamabandiPipeline:
    """
    Complete Jamabandi extraction pipeline
    
    Orchestrates PDF extraction, parsing, LLM enhancement, and Excel export.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize pipeline
        
        Args:
            config_path: Optional path to custom config file
        """
        # Load configuration
        if config_path and config_path.exists():
            from src.core.config import AppConfig, set_config
            config = AppConfig.load_from_yaml(config_path)
            set_config(config)
        
        self.config = get_config()
        
        # Initialize components
        self.camelot_extractor = CamelotExtractor(
            accuracy_threshold=self.config.extraction.camelot_accuracy_threshold,
            line_scale=self.config.extraction.camelot_line_scale
        )
        
        self.column5_parser = Column5Parser()
        self.khasra_splitter = KhasraSplitter(preserve_sum_rows=False)
        self.excel_exporter = ExcelExporter()
        
        # Initialize LLM components if enabled
        self.llm_processor = None
        self.llm_router = None
        
        if self.config.llm.enabled:
            try:
                self.llm_processor = LLMProcessor()
                self.llm_router = LLMRouter(self.llm_processor)
                logger.info("LLM integration enabled")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}. Continuing without LLM support.")
        
        # Performance tracking
        self.perf_logger = PerformanceLogger(logger)
        
        logger.info("Jamabandi Pipeline initialized successfully")
    
    def process(
        self,
        pdf_path: Path,
        output_path: Path,
        pages: str = "all"
    ) -> VillageJamabandi:
        """
        Process a Jamabandi PDF end-to-end
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output Excel file
            pages: Pages to process ("all", "1", "1-5", "1,2,3")
            
        Returns:
            VillageJamabandi with all extracted records
        """
        with log_context(pdf_path=str(pdf_path), pages=pages):
            logger.info(f"Starting processing: {pdf_path}")
            
            try:
                # Step 1: Extract metadata
                with self.perf_logger.track("extract_metadata"):
                    metadata = self._extract_metadata(pdf_path)
                    logger.info(f"Village: {metadata.babata_mouza}, District: {metadata.zla}")
                
                # Step 2: Extract tables
                with self.perf_logger.track("extract_tables"):
                    tables, fallback_pages = self.camelot_extractor.extract_with_fallback_info(
                        str(pdf_path),
                        pages=pages
                    )
                    
                    if not tables:
                        raise ValueError("No tables extracted from PDF")
                    
                    logger.info(f"Extracted {len(tables)} tables")
                    if fallback_pages:
                        logger.warning(f"Pages needing fallback: {fallback_pages}")
                
                # Step 3: Process each table
                all_records = []
                
                for table in tables:
                    with self.perf_logger.track(f"process_table_page_{table.page}"):
                        logger.info(f"Processing page {table.page} (accuracy: {table.accuracy:.1f}%)")
                        records = self._process_table(table, table.page)
                        all_records.extend(records)
                        logger.info(f"Extracted {len(records)} records from page {table.page}")
                
                # Step 4: Create village data
                village_data = VillageJamabandi(
                    metadata=metadata,
                    records=all_records,
                    source_file=pdf_path.name,
                    total_pages=len(tables)
                )
                
                # Step 5: Export to Excel
                with self.perf_logger.track("export_excel"):
                    self.excel_exporter.export(village_data, output_path)
                
                # Log summary
                self._log_summary(village_data, output_path)
                self.perf_logger.log_summary()
                
                return village_data
                
            except Exception as e:
                logger.error(f"Processing failed: {e}", exc_info=True)
                raise
    
    def _extract_metadata(self, pdf_path: Path) -> VillageMetadata:
        """Extract village metadata from PDF header"""
        import fitz
        
        doc = fitz.open(str(pdf_path))
        first_page_text = doc[0].get_text()
        doc.close()
        
        metadata = {
            'sal': '2017-18',
            'zla': 'Unknown',
            'thsil': 'Unknown',
            'babata_mouza': pdf_path.stem
        }
        
        # Parse header
        lines = first_page_text.split('\n')
        for line in lines[:10]:
            line_lower = line.lower()
            
            if ':sal' in line_lower or ': sal' in line_lower:
                import re
                year_match = re.search(r'(\d{4}-\d{2})', line)
                if year_match:
                    metadata['sal'] = year_match.group(1)
            
            if ':zla' in line_lower:
                parts = line.split(':zla')
                if parts:
                    district = parts[0].strip().split()[-1] if parts[0].strip() else 'Unknown'
                    metadata['zla'] = district
            
            if ':thsil' in line_lower:
                parts = line.split(':thsil')
                if parts:
                    tehsil = parts[0].strip().split(':')[-1].strip() if parts[0].strip() else 'Unknown'
                    metadata['thsil'] = tehsil
            
            if ':babata' in line_lower:
                parts = line.lower().split(':babata')
                if parts:
                    village = parts[0].strip().split(':')[-1].strip() if parts[0].strip() else pdf_path.stem
                    metadata['babata_mouza'] = village.title()
        
        return VillageMetadata(**metadata)
    
    def _process_table(self, table, page_number: int) -> list[JamabandiRecord]:
        """Process extracted table into records"""
        df = table.dataframe
        records = []
        
        # Identify columns
        column_map = self._identify_columns(df)
        
        for row_idx, row in df.iterrows():
            try:
                # Skip header rows
                if self._is_header_row(row):
                    continue
                
                # Extract basic fields
                khevat = self._safe_int(row.iloc[column_map.get('khevat', 0)])
                khata = self._safe_int(row.iloc[column_map.get('khata', 1)])
                
                # If khata is 0, it might be a continuation row or invalid
                # But we should check if it has other data
                
                # Parse cultivator info (Column 5) with LLM fallback
                cultivator_text = str(row.iloc[column_map.get('cultivator', 4)])
                cultivator_info = self._parse_cultivator_with_fallback(cultivator_text)
                
                # Parse Khasra
                khasra_hal = str(row.iloc[column_map.get('khasra_hal', 6)])
                khasra_sabik = str(row.iloc[column_map.get('khasra_sabik', 7)]) if 'khasra_sabik' in column_map else None
                
                # Combine Khasra if split
                if khasra_sabik and khasra_sabik.strip():
                    khasra_text = f"{khasra_hal}/{khasra_sabik}"
                else:
                    khasra_text = khasra_hal
                
                khasras = self.khasra_splitter.parse_khasra_numbers(khasra_text)
                
                # Parse Area
                area_kanal = self._safe_int(row.iloc[column_map.get('area_kanal', 0)]) if 'area_kanal' in column_map else 0
                area_marla = self._safe_int(row.iloc[column_map.get('area_marla', 0)]) if 'area_marla' in column_map else 0
                area_type = str(row.iloc[column_map.get('area_type', 0)]) if 'area_type' in column_map else None
                
                # Handle combined area column if not split
                if 'area' in column_map:
                    area_text = str(row.iloc[column_map.get('area')])
                    areas = self.khasra_splitter.parse_areas(area_text)
                    # Use parsed area if available
                    if areas:
                        area_obj = areas[0]
                else:
                    area_obj = AreaInfo(
                        kanal=area_kanal,
                        marla=area_marla,
                        kisam_zamin=area_type
                    )

                # Parse Mutation
                mutation_type = str(row.iloc[column_map.get('mutation_type', 0)]) if 'mutation_type' in column_map else None
                mutation_num = str(row.iloc[column_map.get('mutation_num', 0)]) if 'mutation_num' in column_map else None
                
                mutation = None
                if mutation_type or mutation_num:
                    mutation = MutationInfo(
                        kisam=mutation_type,
                        number=mutation_num,
                        raw_text=f"{mutation_type} {mutation_num}".strip()
                    )

                # Skip sum rows
                if self.khasra_splitter.is_sum_row(row):
                    continue
                
                # Create records (one per Khasra)
                for i, khasra in enumerate(khasras):
                    # For split rows, we might need to distribute area or keep it on the first one
                    # For now, assign to the first one or use the specific area object
                    
                    record = JamabandiRecord(
                        number_khevat=khevat,
                        number_khata=khata,
                        nam_tarf_ya_patti=str(row.iloc[column_map.get('nambardar', 2)]) if column_map.get('nambardar') else None,
                        nam_malik_meh_ahval=str(row.iloc[column_map.get('malik', 3)]) if column_map.get('malik') else None,
                        cultivator=cultivator_info,
                        vasayil_abapashi=str(row.iloc[column_map.get('irrigation', 5)]) if column_map.get('irrigation') else None,
                        khasra=KhasraInfo(hal=khasra.hal, sabik=khasra.sabik),
                        area=area_obj,
                        mutation=mutation,
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

    def _parse_cultivator_with_fallback(self, text: str) -> CultivatorInfo:
        """Parse cultivator field with LLM fallback"""
        # Try rule-based first
        result = self.column5_parser.parse(text)
        
        # Check if LLM fallback needed
        if self.llm_router and result.confidence < self.config.llm.confidence_threshold:
            try:
                llm_result = self.llm_processor.parse_cultivator_field(text)
                
                # Convert LLM result to CultivatorInfo
                return CultivatorInfo(
                    name=llm_result.get('cultivator_name', ''),
                    parentage_type=llm_result.get('parentage_type'),
                    parentage_name=llm_result.get('parentage_name'),
                    caste=llm_result.get('caste'),
                    residence=llm_result.get('residence'),
                    ownership_type=llm_result.get('ownership_type'),
                    share_fraction=llm_result.get('share_fraction'),
                    role=llm_result.get('role', 'cultivator'),
                    raw_text=text,
                    confidence=0.9,
                    extraction_method=ExtractionMethod.LLM
                )
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
        
        return result

    def _identify_columns(self, df) -> dict:
        """Identify column positions"""
        num_cols = len(df.columns)
        
        if num_cols == 16:
            # 16-column layout (Reverse order 15->0)
            return {
                'khevat': 15,          # Col 1
                'khata': 14,           # Col 2
                'nambardar': 13,       # Col 3
                'malik': 12,           # Col 4
                'cultivator': 11,      # Col 5
                'irrigation': 10,      # Col 6
                'khasra_sabik': 9,     # Col 7 (Split)
                'khasra_hal': 8,       # Col 7 (Split)
                'area_kanal': 7,       # Col 8 (Split)
                'area_marla': 6,       # Col 8 (Split)
                'area_type': 5,        # Col 8 (Split)
                'rent': 4,             # Col 9
                'cess': 3,             # Col 10
                'mutation_num': 2,     # Col 11 (Split)
                'mutation_type': 1,    # Col 11 (Split)
                'remarks': 0           # Col 12
            }
        elif num_cols >= 12:
            return {
                'remarks': 0, 'mutation': 1, 'cess': 2, 'tax': 3,
                'area': 4, 'khasra': 5, 'irrigation': 6, 'cultivator': 7,
                'malik': 8, 'nambardar': 9, 'khata': 10, 'khevat': 11,
            }
        else:
            logger.warning(f"Only {num_cols} columns found, using simplified mapping")
            return {
                'khevat': num_cols - 1,
                'khata': num_cols - 2 if num_cols > 1 else 0,
                'cultivator': min(4, num_cols - 1),
                'khasra': min(5, num_cols - 1),
                'area': min(6, num_cols - 1),
            }
    
    def _is_header_row(self, row) -> bool:
        """Check if row is a header"""
        header_keywords = [
            'number khata', 'number khevat', 'kaifiyat', 'nam malik',
            'nam kashtakar', 'khasra', 'raqba', 'vasayil'
        ]
        
        row_text = ' '.join(str(v).lower() for v in row.values)
        return any(keyword in row_text for keyword in header_keywords)
    
    def _safe_int(self, value) -> int:
        """Safely convert to int"""
        try:
            import pandas as pd
            if pd.isna(value):
                return 0
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return 0
    
    def _log_summary(self, village_data: VillageJamabandi, output_path: Path):
        """Log processing summary"""
        summary = village_data.get_summary()
        
        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info(f"  Village: {village_data.metadata.babata_mouza}")
        logger.info(f"  Records: {summary['total_records']}")
        logger.info(f"  Avg Confidence: {summary['average_confidence']:.1%}")
        logger.info(f"  Needs Review: {summary['records_needing_review']} ({summary['review_percentage']:.1f}%)")
        logger.info(f"  Output: {output_path}")
        logger.info("=" * 60)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='J&K Jamabandi Land Record Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input PDF path')
    parser.add_argument('--output', '-o', required=True, help='Output Excel path')
    parser.add_argument('--pages', '-p', default='all', help='Pages to process (default: all)')
    parser.add_argument('--config', '-c', help='Path to config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        from src.core.logger import setup_logging
        setup_logging(config_override={'level': 'DEBUG'})
    
    # Determine output path
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if output_path.is_dir():
        output_path = output_path / f"{input_path.stem}_output.xlsx"
    
    # Run pipeline
    try:
        config_path = Path(args.config) if args.config else None
        pipeline = JamabandiPipeline(config_path=config_path)
        
        result = pipeline.process(
            pdf_path=input_path,
            output_path=output_path,
            pages=args.pages
        )
        
        print(f"\n✅ Processing complete!")
        print(f"   Records: {result.total_records}")
        print(f"   Output: {output_path}")
        
    except AgriStackError as e:
        logger.error(f"Processing failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
