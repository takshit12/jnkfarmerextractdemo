
"""
Debug Extraction Logic

Runs the pipeline on specific pages with enhanced logging to trace data extraction.
"""
import sys
from pathlib import Path
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path.cwd()))

from src.main import JamabandiPipeline
from src.core.config import set_config, AppConfig

def debug_extraction():
    pdf_path = Path("TransliteradVersion_Village Gujral - Jamabandi (1).pdf")
    output_path = Path("output/debug_output.xlsx")
    
    # Initialize pipeline
    pipeline = JamabandiPipeline()
    
    print(f"Processing {pdf_path}...")
    
    # Extract tables manually to inspect dataframe
    tables, _ = pipeline.camelot_extractor.extract_with_fallback_info(
        str(pdf_path),
        pages="1"
    )
    
    if tables:
        df = tables[0].dataframe
        print(f"\nExtracted DF Shape: {df.shape}")
        
        # Print column 10 specifically (Irrigation)
        print("\n--- Column 10 (Irrigation) Raw Values ---")
        if 10 < len(df.columns):
            print(df.iloc[:, 10].head(10).to_string())
        else:
            print("Column 10 not found!")
            
        # Run process_table to see mapping
        print("\n--- Running _identify_columns ---")
        col_map = pipeline._identify_columns(df)
        print(f"Mapping: {col_map}")
        
        print(f"Irrigation index details: Map says {col_map.get('irrigation')}")
        
        # Process rows and check output
        print("\n--- Processing Rows ---")
        
        # Dump Row 3 raw
        print("\n--- Row 3 Raw Content ---")
        if len(df) > 3:
            row3 = df.iloc[3]
            for i, val in enumerate(row3):
                print(f"Col {i}: '{val}'")
        
        records = pipeline._process_table(tables[0], 1)
        
        print("\n--- Extracted Records (First 3) ---")
        for rec in records[:3]:
            print(f"Khevat: {rec.number_khevat}, Khata: {rec.number_khata}")
            print(f"  Cultivator: {rec.cultivator.name}")
            print(f"  Irrigation: {rec.vasayil_abapashi}")
            print(f"  Khasra: {rec.khasra.hal}")
            
    else:
        print("No tables extracted!")

if __name__ == "__main__":
    debug_extraction()
