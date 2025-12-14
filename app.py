"""
AgriStack Land Record Digitization - Web UI

Beautiful Streamlit interface for PDF processing.
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import time
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import JamabandiPipeline
from src.core import get_logger, get_config
from src.core.exceptions import AgriStackError

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="AgriStack - Land Record Digitization",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'output_file' not in st.session_state:
        st.session_state.output_file = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0


def render_header():
    """Render page header"""
    st.markdown('<div class="main-header">🏛️ AgriStack Land Record Digitization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transform Jamabandi PDFs into structured Excel data</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render sidebar with configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        config = get_config()
        
        # LLM Settings
        st.subheader("🤖 LLM Settings")
        llm_enabled = st.checkbox(
            "Enable LLM Enhancement",
            value=config.llm.enabled,
            help="Use AI to parse complex fields"
        )
        
        if llm_enabled:
            provider = st.selectbox(
                "Provider",
                ["openai", "anthropic"],
                index=0 if config.llm.provider == "openai" else 1
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config.llm.confidence_threshold,
                step=0.05,
                help="Minimum confidence before using LLM"
            )
        
        st.markdown("---")
        
        # Extraction Settings
        st.subheader("📄 Extraction Settings")
        accuracy_threshold = st.slider(
            "Camelot Accuracy Threshold",
            min_value=50.0,
            max_value=100.0,
            value=config.extraction.camelot_accuracy_threshold,
            step=5.0,
            help="Minimum table extraction accuracy"
        )
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.info("""
        **AgriStack v1.0.0**
        
        Automated extraction of land records from Jamabandi PDFs for the AgriStack initiative.
        
        **Features:**
        - Multi-method PDF extraction
        - AI-powered parsing
        - LRIS-compliant output
        - Confidence scoring
        """)
        
        return {
            'llm_enabled': llm_enabled,
            'provider': provider if llm_enabled else None,
            'confidence_threshold': confidence_threshold if llm_enabled else None,
            'accuracy_threshold': accuracy_threshold
        }


def process_pdf(uploaded_file, settings):
    """Process uploaded PDF"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_input:
            tmp_input.write(uploaded_file.getvalue())
            input_path = Path(tmp_input.name)
        
        output_path = Path(tempfile.gettempdir()) / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Initialize pipeline
        pipeline = JamabandiPipeline()
        
        # Update config if needed
        if settings['llm_enabled'] is not None:
            config = get_config()
            config.llm.enabled = settings['llm_enabled']
            if settings['provider']:
                config.llm.provider = settings['provider']
            if settings['confidence_threshold']:
                config.llm.confidence_threshold = settings['confidence_threshold']
            config.extraction.camelot_accuracy_threshold = settings['accuracy_threshold']
        
        # Process
        start_time = time.time()
        result = pipeline.process(
            pdf_path=input_path,
            output_path=output_path,
            pages="all"
        )
        processing_time = time.time() - start_time
        
        # Clean up input file
        input_path.unlink()
        
        return result, output_path, processing_time
        
    except AgriStackError as e:
        st.error(f"Processing Error: {e}")
        logger.error(f"Processing failed: {e}")
        return None, None, 0
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None, None, 0


def render_results(result, output_file, processing_time):
    """Render processing results"""
    st.success("✅ Processing Complete!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    summary = result.get_summary()
    
    with col1:
        st.metric(
            label="📊 Total Records",
            value=summary['total_records']
        )
    
    with col2:
        st.metric(
            label="📄 Pages Processed",
            value=result.total_pages
        )
    
    with col3:
        st.metric(
            label="🎯 Avg Confidence",
            value=f"{summary['average_confidence']:.1%}"
        )
    
    with col4:
        st.metric(
            label="⏱️ Processing Time",
            value=f"{processing_time:.1f}s"
        )
    
    st.markdown("---")
    
    # Village Information
    st.subheader("🏘️ Village Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Village:**")
        st.write(result.metadata.babata_mouza)
    
    with col2:
        st.write("**District:**")
        st.write(result.metadata.zla)
    
    with col3:
        st.write("**Tehsil:**")
        st.write(result.metadata.thsil)
    
    with col4:
        st.write("**Year:**")
        st.write(result.metadata.sal)
    
    st.markdown("---")
    
    # Quality Metrics
    st.subheader("📈 Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Records Needing Review:**")
        st.write(f"{summary['records_needing_review']} ({summary['review_percentage']:.1f}%)")
        
        if summary['review_percentage'] > 20:
            st.warning("⚠️ High percentage of records need manual review")
        elif summary['review_percentage'] > 10:
            st.info("ℹ️ Some records may need review")
        else:
            st.success("✅ Most records have high confidence")
    
    with col2:
        st.write("**Unique Khata Numbers:**")
        st.write(summary.get('unique_khata', 'N/A'))
        
        st.write("**Unique Khasra Numbers:**")
        st.write(summary.get('unique_khasra', 'N/A'))
    
    st.markdown("---")
    
    # Sample Data Preview
    st.subheader("👀 Data Preview")
    
    # Convert first 10 records to DataFrame
    preview_data = [record.to_flat_dict() for record in result.records[:10]]
    preview_df = pd.DataFrame(preview_data)
    
    # Select key columns for preview
    preview_columns = [
        'number_khevat', 'number_khata', 'cultivator_name',
        'khasra_hal', 'area_kanal', 'area_marla',
        'extraction_confidence', 'needs_review'
    ]
    
    available_columns = [col for col in preview_columns if col in preview_df.columns]
    
    st.dataframe(
        preview_df[available_columns],
        use_container_width=True,
        hide_index=True
    )
    
    st.caption(f"Showing first 10 of {len(result.records)} records")
    
    st.markdown("---")
    
    # Download Button
    st.subheader("💾 Download Results")
    
    with open(output_file, 'rb') as f:
        excel_data = f.read()
    
    st.download_button(
        label="📥 Download Excel File",
        data=excel_data,
        file_name=f"jamabandi_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


def main():
    """Main application"""
    initialize_session_state()
    render_header()
    
    # Sidebar configuration
    settings = render_sidebar()
    
    # Main content
    st.header("📤 Upload Jamabandi PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a Jamabandi land record PDF document"
    )
    
    if uploaded_file is not None:
        # File info
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Filename:**", uploaded_file.name)
        with col2:
            st.write("**Size:**", f"{uploaded_file.size / 1024:.1f} KB")
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Processing PDF... This may take a few minutes."):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Extracting tables from PDF...")
                progress_bar.progress(25)
                
                # Process
                result, output_file, processing_time = process_pdf(uploaded_file, settings)
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                if result:
                    st.session_state.processed = True
                    st.session_state.result = result
                    st.session_state.output_file = output_file
                    st.session_state.processing_time = processing_time
    
    # Show results if processed
    if st.session_state.processed and st.session_state.result:
        st.markdown("---")
        render_results(
            st.session_state.result,
            st.session_state.output_file,
            st.session_state.processing_time
        )


if __name__ == "__main__":
    main()
