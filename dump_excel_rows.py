
import pandas as pd
import sys

excel_path = r"c:\jnkDocExtractor\jnkfarmerextractdemo\jamabandi_output_20251214_031112.xlsx"
output_txt = r"c:\jnkDocExtractor\jnkfarmerextractdemo\excel_dump.txt"

try:
    df = pd.read_excel(excel_path)
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"Total Rows: {len(df)}\n")
        f.write(f"Columns: {df.columns.tolist()}\n\n")
        
        f.write("--- First 20 Rows ---\n")
        # specific columns of interest to check alignment
        cols = ['number_khevat', 'number_khata', 'nam_tarf_ya_patti', 'cultivator_name', 'cultivator_parentage', 'khasra_hal', 'area_kanal', 'area_marla', 'vasayil_abapashi']
        
        for idx, row in df.head(20).iterrows():
            f.write(f"Row {idx}:\n")
            for col in cols:
                val = str(row.get(col, ''))
                # trunc if too long
                if len(val) > 100: val = val[:100] + "..."
                f.write(f"  {col}: {val}\n")
            f.write("\n")
            
    print(f"Dumped to {output_txt}")

except Exception as e:
    print(f"Error: {e}")
