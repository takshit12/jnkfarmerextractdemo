"""
Analyze Jamabandi PDF Structure

Extracts tables from the first page and prints structure for verification.
"""

import camelot
import pandas as pd
import sys
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

def analyze_pdf(pdf_path):
    print(f"Analyzing: {pdf_path}")
    
    # 1. Extract with Camelot (Lattice)
    print("\n--- Camelot Extraction (Lattice) ---")
    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages='1',
            flavor='lattice',
            line_scale=40
        )
        
        print(f"Found {len(tables)} tables")
        
        if len(tables) > 0:
            df = tables[0].df
            print(f"Table Shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())
            
            print("\nColumn Analysis:")
            for i, col in enumerate(df.columns):
                sample_val = df.iloc[1][col] if len(df) > 1 else "N/A"
                header_val = df.iloc[0][col]
                print(f"Col {i}: Header='{header_val}' | Sample='{sample_val}'")
                
    except Exception as e:
        print(f"Camelot error: {e}")

    # 2. Extract with pdfplumber (for text comparison)
    print("\n--- pdfplumber Extraction ---")
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            text = page.extract_text()
            print("Page Text Preview (first 500 chars):")
            print(text[:500])
            
            print("\nTable Extraction:")
            tables = page.extract_tables()
            if tables:
                print(f"Found {len(tables)} tables")
                df = pd.DataFrame(tables[0])
                print(df.head())
    except Exception as e:
        print(f"pdfplumber error: {e}")

if __name__ == "__main__":
    pdf_path = r"c:\jnkDocExtractor\jnkfarmerextractdemo\TransliteradVersion_Village Gujral - Jamabandi (1).pdf"
    analyze_pdf(pdf_path)
