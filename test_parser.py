
from column5_parser import Column5Parser

try:
    print("Initializing parser...")
    parser = Column5Parser()
    print("Parser initialized.")
    
    text = "kasht bajariye mulazam sahid singh pisar attar singh"
    print(f"Parsing: {text}")
    result = parser.parse(text)
    print("Result:", result)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
