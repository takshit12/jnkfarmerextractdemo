#!/usr/bin/env python
"""
Setup script for AgriStack Land Record Digitization

Automates environment setup and dependency installation.
"""

import subprocess
import sys
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Run shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ghostscript():
    """Check if Ghostscript is installed"""
    print("\nChecking Ghostscript...")
    if run_command("gs --version", check=False):
        print("✅ Ghostscript is installed")
        return True
    else:
        print("❌ Ghostscript not found")
        print("   Download from: https://ghostscript.com/releases/gsdnld.html")
        return False


def create_venv():
    """Create virtual environment"""
    print("\nCreating virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("Recreate? (y/N): ")
        if response.lower() != 'y':
            return True
        
        import shutil
        shutil.rmtree(venv_path)
    
    if run_command(f"{sys.executable} -m venv venv"):
        print("✅ Virtual environment created")
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling dependencies...")
    
    # Determine pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    # Upgrade pip
    run_command(f"{pip_path} install --upgrade pip")
    
    # Install requirements
    if run_command(f"{pip_path} install -r requirements.txt"):
        print("✅ Dependencies installed")
        return True
    else:
        print("❌ Failed to install dependencies")
        return False


def create_env_file():
    """Create .env file template"""
    print("\nCreating .env file...")
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️  .env file already exists")
        return True
    
    env_template = """# AgriStack Configuration

# LLM API Keys (choose one)
OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Override config settings
# LLM__PROVIDER=openai
# LLM__MODEL=gpt-4o
# LLM__CONFIDENCE_THRESHOLD=0.7
"""
    
    env_path.write_text(env_template)
    print("✅ .env file created")
    print("   Please edit .env and add your API keys")
    return True


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    dirs = [
        "logs",
        "output",
        ".cache",
        "data/input",
        "data/output",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True


def main():
    """Main setup function"""
    print("=" * 60)
    print("AgriStack Land Record Digitization - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check Ghostscript
    gs_installed = check_ghostscript()
    
    # Create virtual environment
    if not create_venv():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Create directories
    create_directories()
    
    # Final summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    if not gs_installed:
        print("\n⚠️  WARNING: Ghostscript not installed")
        print("   Camelot extraction will not work without it")
        print("   Download from: https://ghostscript.com/releases/gsdnld.html")
    
    print("\nNext steps:")
    print("1. Edit .env and add your API keys")
    
    if platform.system() == "Windows":
        print("2. Activate virtual environment: venv\\Scripts\\activate")
    else:
        print("2. Activate virtual environment: source venv/bin/activate")
    
    print("3. Run: python -m src.main --input sample.pdf --output output.xlsx")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
