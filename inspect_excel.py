
import pandas as pd
import sys

excel_path = r"c:\jnkDocExtractor\jnkfarmerextractdemo\jamabandi_output_20251214_031112.xlsx"

try:
    df = pd.read_excel(excel_path)
    print(f"Total Rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    print("\n--- First 10 Rows ---")
    print(df.head(10).to_string())
    
    print("\n--- Rows with 'Needs Review' ---")
    review_df = df[df['needs_review'] == True]
    print(f"Total Needs Review: {len(review_df)}")
    if not review_df.empty:
        print(review_df.head(5).to_string())
        
except Exception as e:
    print(f"Error reading Excel: {e}")
