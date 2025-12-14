import pandas as pd
import sys

try:
    df = pd.read_excel("output/gujral_output_p1.xlsx")
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
except Exception as e:
    print(f"Error reading Excel: {e}")
