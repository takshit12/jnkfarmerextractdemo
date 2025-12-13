# Cursor AI Instructions

## 🎯 Project Context

You are helping develop a **PDF data extraction system** for Jamabandi (land record) documents from Jammu & Kashmir, India. This is for the **AgriStack** initiative - building a digital farmer and farm registry.

### The Core Challenge

1. **Input**: PDF files containing land records in transliterated Urdu (Roman script)
2. **Output**: Structured Excel files matching the LRIS (Land Record Information System) schema
3. **Key Difficulty**: Semi-structured text parsing, multi-value cells, 12-column table extraction

---

## 📚 Domain Knowledge

### What is a Jamabandi?

A Jamabandi is an official government land record document containing:
- **Khevat**: Subdivision of a revenue village (Column 1)
- **Khata**: Account number within Khevat (Column 2) - PRIMARY ROW DELIMITER
- **Cultivator Info**: Name, parentage, caste, village (Column 5) - NEEDS PARSING
- **Khasra**: Survey number of land parcel (Column 7) - NEEDS SPLITTING
- **Area**: Land measurement in Kanal/Marla (Column 8)
- **Mutation**: Transaction history (Column 11)

### Transliterated Urdu Terms

These keywords are CRITICAL for parsing Column 5:

| Term | Meaning | Use |
|------|---------|-----|
| `kasht` | cultivator | Marks start of name |
| `pisar` | son of | Parentage marker |
| `pisaran` | sons of | Parentage marker (plural) |
| `dukhtar` | daughter of | Parentage marker |
| `dukhtaran` | daughters of | Parentage marker (plural) |
| `zoja` | wife of | Parentage marker |
| `byuh` | widow of | Parentage marker |
| `kaum` | caste | Caste marker |
| `sakin` | resident | Residence marker |
| `sakindeh` | resident of same village | Residence marker |
| `morosi` | hereditary | Ownership type |
| `gair morosi` | non-heirs | Ownership type |
| `alati` | temporary | Ownership type |
| `kitta` | total (sum row) | Aggregation marker |

### Example Column 5 Text

```
kasht sahid v singh pisar attar singh kaum sukh sakindeh gair morosi
```

Parse into:
- **Name**: sahid v singh
- **Parentage**: S/o attar singh
- **Caste**: sukh
- **Residence**: sakindeh (same village)
- **Ownership**: gair morosi (non-heirs)

---

## 🏗️ Architecture Overview

```
PDF Input
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 1: PDF Extraction            │
│  - Camelot (primary, lattice mode)  │
│  - pdfplumber (fallback)            │
│  - PyMuPDF (metadata)               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 2: Jamabandi Parsing         │
│  - Column 5 Parser (NER-style)      │
│  - Khasra Splitter (expand rows)    │
│  - Row Delimiter by Khata           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 3: LLM Processing            │
│  - GPT-4o / Claude for complex cells│
│  - Confidence-based routing         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 4: Validation & Export       │
│  - Pydantic models                  │
│  - LRIS schema compliance           │
│  - Excel output                     │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
jk-agristack-landrecords/
├── src/
│   ├── extractors/           # PDF extraction modules
│   │   ├── pdf_loader.py     # PyMuPDF-based loader
│   │   ├── camelot_extractor.py
│   │   └── pdfplumber_extractor.py
│   ├── parsers/              # Jamabandi-specific parsing
│   │   ├── jamabandi_parser.py   # Main orchestrator
│   │   ├── column5_parser.py     # Cultivator field parser
│   │   ├── khasra_splitter.py    # Multi-khasra row expander
│   │   └── urdu_transliteration.py
│   ├── processors/           # LLM integration
│   │   ├── llm_processor.py
│   │   └── batch_processor.py
│   ├── validators/           # Data validation
│   │   ├── models.py         # Pydantic models
│   │   └── schema_validator.py
│   └── utils/               # Utilities
│       ├── excel_exporter.py
│       └── deduplicator.py
├── config/
│   ├── settings.yaml
│   └── column_mappings.yaml
├── data/
│   ├── input/               # Place PDFs here
│   └── output/              # Generated Excel files
└── tests/
```

---

## 🔧 Key Implementation Patterns

### Pattern 1: Column 5 Parsing with Regex

```python
def parse_column5(text: str) -> dict:
    """Parse cultivator field using keyword markers"""
    text = text.lower().strip()
    
    result = {
        'name': None,
        'parentage_type': None,
        'parentage_name': None,
        'caste': None,
        'residence': None,
        'ownership_type': None
    }
    
    # Extract name (after 'kasht', before parentage marker)
    parentage_markers = ['pisar', 'pisaran', 'dukhtar', 'dukhtaran', 'zoja', 'byuh']
    
    if 'kasht' in text:
        start = text.index('kasht') + len('kasht')
        end = len(text)
        for marker in parentage_markers:
            if marker in text:
                end = min(end, text.index(marker))
        result['name'] = text[start:end].strip()
    
    # Extract parentage
    for marker in parentage_markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find('kaum', start) if 'kaum' in text[start:] else len(text)
            result['parentage_name'] = text[start:end].strip()
            result['parentage_type'] = {
                'pisar': 'S/o', 'pisaran': 'S/o',
                'dukhtar': 'D/o', 'dukhtaran': 'D/o',
                'zoja': 'W/o', 'byuh': 'Widow/o'
            }.get(marker)
            break
    
    # Extract caste (after 'kaum')
    if 'kaum' in text:
        start = text.index('kaum') + len('kaum')
        end = len(text)
        for marker in ['sakin', 'sakindeh', 'morosi', 'gair', 'alati']:
            if marker in text[start:]:
                end = min(end, start + text[start:].index(marker))
        result['caste'] = text[start:end].strip()
    
    return result
```

### Pattern 2: Khasra Number Splitting

```python
def split_khasra_rows(row: pd.Series) -> List[pd.Series]:
    """Expand row with multiple khasra numbers into separate rows"""
    import re
    
    khasra_text = str(row.get('khasra_numbers', ''))
    area_text = str(row.get('area', ''))
    
    # Skip sum rows (marked with 'kitta')
    if 'kitta' in khasra_text.lower():
        return []  # Or return marked as sum row
    
    # Extract khasra numbers (format: hal or hal/sabik)
    khasra_pattern = r'(\d+)(?:/(\d+))?'
    khasras = re.findall(khasra_pattern, khasra_text)
    
    # Extract areas (format: kanal marla)
    area_pattern = r'(\d+)\s+(\d+)'
    areas = re.findall(area_pattern, area_text)
    
    # Create individual rows
    result = []
    for i, khasra in enumerate(khasras):
        new_row = row.copy()
        new_row['khasra_hal'] = khasra[0]
        new_row['khasra_sabik'] = khasra[1] if khasra[1] else None
        if i < len(areas):
            new_row['area_kanal'] = areas[i][0]
            new_row['area_marla'] = areas[i][1]
        result.append(new_row)
    
    return result
```

### Pattern 3: Confidence-Based LLM Routing

```python
def process_with_fallback(text: str, rule_parser, llm_processor, threshold=0.7):
    """Use rule-based first, fallback to LLM if confidence low"""
    
    # Try rule-based parsing
    result = rule_parser.parse(text)
    
    # Calculate confidence based on fields extracted
    fields = ['name', 'parentage_type', 'caste', 'residence']
    filled = sum(1 for f in fields if result.get(f))
    confidence = filled / len(fields)
    
    if confidence >= threshold:
        result['confidence'] = confidence
        result['method'] = 'rule_based'
        return result
    
    # Fallback to LLM
    llm_result = llm_processor.parse(text)
    llm_result['confidence'] = 0.9  # LLM generally high confidence
    llm_result['method'] = 'llm'
    return llm_result
```

---

## 🎯 Current Development Focus

### Priority 1: Get Basic Extraction Working
- [ ] Camelot table extraction from transliterated PDF
- [ ] Verify 12-column structure is captured
- [ ] Handle page breaks and headers

### Priority 2: Column 5 Parser
- [ ] Implement regex-based extraction
- [ ] Test on 10+ sample entries
- [ ] Handle edge cases (missing fields, unusual formats)

### Priority 3: Khasra Splitting
- [ ] Parse multiple khasra numbers
- [ ] Map areas to correct khasra
- [ ] Handle 'kitta' sum rows

### Priority 4: Integration
- [ ] End-to-end pipeline
- [ ] Excel output matching LRIS schema
- [ ] Confidence scoring and flagging

---

## ⚠️ Common Pitfalls

1. **Camelot Installation**: Requires Ghostscript. If errors, check `gs` is in PATH.

2. **Column Order**: Jamabandi columns are RIGHT-TO-LEFT (1→12). The PDF may show them reversed.

3. **Multi-line Cells**: Cultivator info often spans multiple lines. Join with space before parsing.

4. **Area Format**: Can be "16 0" (16 kanal, 0 marla) or "0 16" (0 kanal, 16 marla). Context matters.

5. **Kitta Rows**: Sum rows marked with 'kitta' should NOT be expanded, they aggregate parcels.

6. **Name Variations**: Same person may appear as "singh" or "singha" or "sn". Normalize.

---

## 🔍 Debugging Tips

### Check Camelot Extraction
```python
import camelot
tables = camelot.read_pdf("input.pdf", pages="1", flavor="lattice")
print(f"Found {len(tables)} tables")
print(f"Accuracy: {tables[0].accuracy}%")
print(tables[0].df)  # View extracted DataFrame
```

### Visualize Table Detection
```python
import camelot
tables = camelot.read_pdf("input.pdf", pages="1", flavor="lattice")
camelot.plot(tables[0], kind='grid').show()
```

### Test Column 5 Parser
```python
test_cases = [
    "kasht sahid v singh pisar attar singh kaum sukh sakindeh gair morosi",
    "kasht sandhu singh pisar sain kaum rakwal sakindeh alati",
    "iqubal singh 1/3 manmohan singh 1/3 harbans singh 1/3 pisaran bachan singh kaum namalum",
]

parser = Column5Parser()
for text in test_cases:
    result = parser.parse(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print("---")
```

---

## 📝 When Generating Code

1. **Always add type hints** - This is a data processing pipeline, types matter.

2. **Use Pydantic for validation** - All data structures should be validated.

3. **Log extensively** - Processing failures need debugging.

4. **Handle edge cases gracefully** - Empty cells, malformed data, missing fields.

5. **Calculate confidence scores** - Flag uncertain extractions for review.

6. **Test incrementally** - Don't try to process 100 pages before 1 page works.

---

## 🔗 Key Dependencies

```
camelot-py[cv]>=0.11.0  # Table extraction (requires ghostscript)
pdfplumber>=0.10.0       # Fallback extraction
pymupdf>=1.23.0          # PDF handling
pandas>=2.0.0            # Data manipulation
openpyxl>=3.1.0          # Excel export
pydantic>=2.0.0          # Data validation
openai>=1.0.0            # GPT-4o integration
anthropic>=0.18.0        # Claude integration
python-dotenv>=1.0.0     # Environment variables
streamlit>=1.30.0        # Demo UI
```

---

## 🏆 Success Criteria

For the hackathon submission (December 15):

1. ✅ Process transliterated PDF end-to-end
2. ✅ Extract all 12 columns correctly
3. ✅ Parse Column 5 into structured fields
4. ✅ Split multi-khasra rows
5. ✅ Output valid LRIS-schema Excel
6. ✅ Achieve >85% extraction accuracy on sample data
7. ✅ Include confidence scoring
8. ✅ Document the approach in policy brief