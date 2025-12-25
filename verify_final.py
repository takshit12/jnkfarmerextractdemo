
import pandas as pd
import sys

excel_path = "output/gujral_output_p1_final_v2.xlsx"

try:
    df = pd.read_excel(excel_path)
    print("Columns:", df.columns.tolist())
    print("\n--- First 5 Rows ---")
    # showing specific relevant columns
    cols = ['number_khevat', 'cultivator_name', 'cultivator_role', 'vasayil_abapashi', 'khasra_hal', 'needs_review']
    # Filter valid columns
    valid_cols = [c for c in cols if c in df.columns]
    print(df[valid_cols].head(5).to_string())
    
    # Check specifically for irrigation values
    print("\n--- Irrigation Values ---")
    print(df['vasayil_abapashi'].tolist())

    # Check for Bajariye (should NOT be in name if parsed correctly)
    print("\n--- Cultivator Names ---")
    print(df['cultivator_name'].tolist())

except Exception as e:
    print(f"Error reading Excel: {e}")
