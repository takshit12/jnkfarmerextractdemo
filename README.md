# AgriStack Land Record Digitization

Production-ready system for extracting structured data from Jamabandi (land record) PDFs and converting them to LRIS-compliant Excel format.

## Features

- **Multi-Method PDF Extraction**: Camelot (primary) + pdfplumber (fallback)
- **Intelligent Parsing**: Rule-based + LLM fallback for complex fields
- **High Accuracy**: Confidence scoring and validation at every step
- **Scalable Architecture**: Modular design with dependency injection
- **Production Ready**: Comprehensive logging, error handling, and monitoring

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd jnkfarmerextractdemo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ghostscript (required for Camelot)
# Download from: https://ghostscript.com/releases/gsdnld.html
```

### Configuration

Create a `.env` file:

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Usage

#### **Option 1: Web UI (Recommended)**

```bash
# Launch the web interface
run_ui.bat  # Windows
# or
./run_ui.sh  # Linux/Mac

# Opens in browser at http://localhost:8501
```

**Features:**
- 📤 Drag-and-drop PDF upload
- ⚙️ Configurable settings (LLM, thresholds)
- 📊 Real-time processing progress
- 📈 Quality metrics dashboard
- 👀 Data preview
- 💾 One-click Excel download

#### **Option 2: Command Line**

```bash
# Basic usage
python -m src.main --input sample.pdf --output output.xlsx

# Process specific pages
python -m src.main --input sample.pdf --output output.xlsx --pages 1-5

#### **Manual Terminal Execution** (The "Hard" Way)

If you prefer typing commands manually in the terminal:

**1. Frontend (Streamlit UI)**
```powershell
cd c:\jnkDocExtractor\jnkfarmerextractdemo
.\venv\Scripts\activate
streamlit run app.py
```

**2. Backend (CLI Pipeline)**
```powershell
cd c:\jnkDocExtractor\jnkfarmerextractdemo
.\venv\Scripts\activate
python -m src.main --input "TransliteradVersion_Village Gujral - Jamabandi (1).pdf" --output "output/manual_run.xlsx"
```
```

## Architecture

```
src/
├── core/               # Core infrastructure
│   ├── config.py      # Configuration management
│   ├── logger.py      # Structured logging
│   └── exceptions.py  # Custom exceptions
├── extractors/        # PDF extraction
│   └── (uses existing camelot_extractor.py)
├── parsers/           # Data parsing
│   └── (uses existing column5_parser.py, khasra_splitter.py)
├── llm/               # LLM integration
│   ├── processor.py   # LLM API wrapper
│   └── router.py      # Confidence-based routing
├── exporters/         # Data export
│   └── excel_exporter.py
└── main.py            # Main pipeline
```

## Configuration

Edit `settings.yaml` or use environment variables:

```yaml
llm:
  enabled: true
  provider: "openai"  # or "anthropic"
  model: "gpt-4o"
  confidence_threshold: 0.7

extraction:
  camelot_accuracy_threshold: 75.0
  pdfplumber_enabled: true

output:
  format: "xlsx"
  highlight_low_confidence: true
```

## Development

```bash
# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

MIT License

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.