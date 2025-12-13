"""
Camelot-based Table Extractor for Jamabandi PDFs

Uses Camelot's Lattice mode for bordered table extraction.
Provides accuracy scoring and fallback signaling.
"""

import camelot
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    """Container for extracted table data"""
    dataframe: pd.DataFrame
    accuracy: float
    page: int
    extraction_method: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    
    @property
    def needs_fallback(self) -> bool:
        """Check if accuracy is below threshold requiring fallback"""
        return self.accuracy < 75.0
    
    @property
    def row_count(self) -> int:
        return len(self.dataframe)
    
    @property
    def column_count(self) -> int:
        return len(self.dataframe.columns)


class CamelotExtractor:
    """
    Table extractor using Camelot library.
    
    Optimized for Jamabandi documents with:
    - Lattice mode for bordered tables
    - Adjusted line_scale for typical line thickness
    - Vertical text copy for spanning cells
    """
    
    # Jamabandi documents have 12 columns
    EXPECTED_COLUMNS = 12
    
    def __init__(
        self,
        accuracy_threshold: float = 75.0,
        line_scale: int = 40,
        flavor: str = "lattice"
    ):
        """
        Initialize the extractor.
        
        Args:
            accuracy_threshold: Minimum accuracy to consider extraction successful
            line_scale: Scale for line detection (increase for thinner lines)
            flavor: "lattice" for bordered tables, "stream" for borderless
        """
        self.accuracy_threshold = accuracy_threshold
        self.line_scale = line_scale
        self.flavor = flavor
        
        # Validate Ghostscript availability
        self._check_ghostscript()
    
    def _check_ghostscript(self) -> None:
        """Check if Ghostscript is available (required by Camelot)"""
        import shutil
        if not shutil.which('gs') and not shutil.which('gswin64c'):
            logger.warning(
                "Ghostscript not found in PATH. Camelot may fail. "
                "Install Ghostscript: https://ghostscript.com/"
            )
    
    def extract_tables(
        self,
        pdf_path: str,
        pages: str = "all",
        table_areas: Optional[List[str]] = None
    ) -> List[ExtractedTable]:
        """
        Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pages: Page specification ("all", "1", "1,2,3", "1-5")
            table_areas: Optional list of table areas as "x1,y1,x2,y2"
            
        Returns:
            List of ExtractedTable objects
        """
        pdf_path = str(Path(pdf_path).absolute())
        logger.info(f"Extracting tables from: {pdf_path}")
        
        try:
            # Configure Camelot parameters
            kwargs = {
                'pages': pages,
                'flavor': self.flavor,
                'line_scale': self.line_scale,
                'copy_text': ['v'],  # Copy text vertically in spanning cells
                'strip_text': '\n',
            }
            
            if table_areas:
                kwargs['table_areas'] = table_areas
            
            # Extract tables
            tables = camelot.read_pdf(pdf_path, **kwargs)
            
            logger.info(f"Found {len(tables)} tables")
            
            # Convert to our format
            results = []
            for i, table in enumerate(tables):
                df = table.df
                
                # Log extraction details
                logger.debug(
                    f"Table {i+1}: Page {table.page}, "
                    f"Accuracy {table.accuracy:.1f}%, "
                    f"Shape {df.shape}"
                )
                
                # Warn if column count doesn't match expected
                if len(df.columns) != self.EXPECTED_COLUMNS:
                    logger.warning(
                        f"Table {i+1} has {len(df.columns)} columns, "
                        f"expected {self.EXPECTED_COLUMNS}"
                    )
                
                # Create result object
                result = ExtractedTable(
                    dataframe=df,
                    accuracy=table.accuracy,
                    page=table.page,
                    extraction_method="camelot_lattice",
                    bbox=table._bbox if hasattr(table, '_bbox') else None
                )
                
                results.append(result)
                
                # Log if needs fallback
                if result.needs_fallback:
                    logger.warning(
                        f"Table {i+1} accuracy {result.accuracy:.1f}% "
                        f"below threshold {self.accuracy_threshold}%"
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            raise
    
    def extract_single_page(
        self,
        pdf_path: str,
        page_number: int
    ) -> Optional[ExtractedTable]:
        """
        Extract table from a single page.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            ExtractedTable or None if extraction fails
        """
        results = self.extract_tables(pdf_path, pages=str(page_number))
        
        if results:
            return results[0]
        return None
    
    def extract_with_fallback_info(
        self,
        pdf_path: str,
        pages: str = "all"
    ) -> Tuple[List[ExtractedTable], List[int]]:
        """
        Extract tables and return list of pages needing fallback.
        
        Args:
            pdf_path: Path to PDF file
            pages: Page specification
            
        Returns:
            Tuple of (extracted_tables, fallback_pages)
        """
        results = self.extract_tables(pdf_path, pages)
        
        fallback_pages = [
            r.page for r in results 
            if r.needs_fallback
        ]
        
        if fallback_pages:
            logger.info(f"Pages needing fallback: {fallback_pages}")
        
        return results, fallback_pages
    
    @staticmethod
    def visualize_extraction(
        pdf_path: str,
        page: int = 1,
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualize table detection for debugging.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number to visualize
            output_path: Optional path to save visualization
        """
        tables = camelot.read_pdf(pdf_path, pages=str(page), flavor="lattice")
        
        if not tables:
            logger.warning("No tables found to visualize")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig = camelot.plot(tables[0], kind='grid')
            
            if output_path:
                fig.savefig(output_path)
                logger.info(f"Visualization saved to: {output_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.error("matplotlib required for visualization")
    
    def get_extraction_report(
        self,
        results: List[ExtractedTable]
    ) -> dict:
        """
        Generate extraction quality report.
        
        Args:
            results: List of extracted tables
            
        Returns:
            Report dictionary with statistics
        """
        if not results:
            return {'error': 'No tables extracted'}
        
        accuracies = [r.accuracy for r in results]
        
        return {
            'total_tables': len(results),
            'total_rows': sum(r.row_count for r in results),
            'average_accuracy': sum(accuracies) / len(accuracies),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'tables_below_threshold': sum(1 for a in accuracies if a < self.accuracy_threshold),
            'pages_processed': list(set(r.page for r in results)),
        }


# === Alternative: Stream Mode Extractor ===

class CamelotStreamExtractor(CamelotExtractor):
    """
    Stream mode extractor for tables without clear borders.
    
    Uses whitespace analysis instead of line detection.
    Generally lower accuracy but works on more document types.
    """
    
    def __init__(
        self,
        accuracy_threshold: float = 70.0,
        edge_tol: int = 50,
        row_tol: int = 2
    ):
        super().__init__(accuracy_threshold=accuracy_threshold)
        self.flavor = "stream"
        self.edge_tol = edge_tol
        self.row_tol = row_tol
    
    def extract_tables(
        self,
        pdf_path: str,
        pages: str = "all",
        table_areas: Optional[List[str]] = None
    ) -> List[ExtractedTable]:
        """Extract using stream mode with whitespace analysis"""
        
        pdf_path = str(Path(pdf_path).absolute())
        logger.info(f"Extracting (stream mode) from: {pdf_path}")
        
        try:
            kwargs = {
                'pages': pages,
                'flavor': 'stream',
                'edge_tol': self.edge_tol,
                'row_tol': self.row_tol,
            }
            
            if table_areas:
                kwargs['table_areas'] = table_areas
            
            tables = camelot.read_pdf(pdf_path, **kwargs)
            
            results = []
            for table in tables:
                results.append(ExtractedTable(
                    dataframe=table.df,
                    accuracy=table.accuracy,
                    page=table.page,
                    extraction_method="camelot_stream"
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Stream extraction failed: {e}")
            raise


# === Testing ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python camelot_extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Initialize extractor
    extractor = CamelotExtractor(accuracy_threshold=75.0)
    
    # Extract tables
    print(f"\nExtracting tables from: {pdf_path}")
    print("=" * 60)
    
    results, fallback_pages = extractor.extract_with_fallback_info(pdf_path, pages="1")
    
    for result in results:
        print(f"\nPage {result.page}:")
        print(f"  Accuracy: {result.accuracy:.1f}%")
        print(f"  Shape: {result.dataframe.shape}")
        print(f"  Needs fallback: {result.needs_fallback}")
        print(f"\nFirst few rows:")
        print(result.dataframe.head())
    
    # Generate report
    report = extractor.get_extraction_report(results)
    print("\n" + "=" * 60)
    print("Extraction Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")