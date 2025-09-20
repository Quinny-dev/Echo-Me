#!/bin/bash
# ASL Gloss to English Translation System - Fixed Installation Script
# 
# This script automates the complete setup process:
# 1. Creates directory structure
# 2. Sets up Python environment
# 3. Installs dependencies  
# 4. Generates training data
# 5. Trains the model
# 6. Tests the system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
PROJECT_NAME="asl-gloss-translator"
PYTHON_VERSION="3.8"
VENV_NAME="asl_env"

print_banner() {
    echo ""
    echo "🚀 ASL Gloss to English Translation System"
    echo "==========================================="
    echo "Lightweight NLP system using T5"
    echo "Supports sentences and paragraphs"
    echo "Mobile-ready with quantization"
    echo ""
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found. Please install Python 3.8+."
        exit 1
    fi
    
    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Found Python $PYTHON_VER"
    
    # Check pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        log_error "pip not found. Please install pip."
        exit 1
    fi
    
    # Check if we have the required files
    if [ ! -f "data/synthetic_dataset.py" ]; then
        log_error "Missing data/synthetic_dataset.py"
        exit 1
    fi
    
    if [ ! -f "src/train.py" ]; then
        log_error "Missing src/train.py"
        exit 1
    fi
    
    if [ ! -f "requirements.txt" ]; then
        log_error "Missing requirements.txt"
        exit 1
    fi
    
    log_success "All required files found"
    log_success "System requirements check passed"
}

setup_environment() {
    log_info "Setting up Python virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_NAME" ]; then
        $PYTHON_CMD -m venv $VENV_NAME
        log_success "Created virtual environment: $VENV_NAME"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source $VENV_NAME/bin/activate
    log_success "Activated virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    log_success "Updated pip and build tools"
}

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Install from requirements.txt
    pip install -r requirements.txt
    
    log_success "Dependencies installed successfully"
}

generate_training_data() {
    log_info "Generating synthetic training dataset..."
    
    # Navigate to data directory and run the script
    cd data
    
    # Run the dataset generation script (it's in the current directory)
    $PYTHON_CMD synthetic_dataset.py
    
    # Go back to root directory
    cd ..
    
    log_success "Training dataset generated"
}

train_model() {
    log_info "Training ASL translation model..."
    log_info "This may take 1-3 hours depending on your hardware..."
    
    cd src
    
    # Start training with default parameters
    $PYTHON_CMD train.py \
        --model_name "google/t5-efficient-tiny" \
        --num_epochs 3 \
        --batch_size 8 \
        --learning_rate 3e-4 \
        --device auto
    
    cd ..
    log_success "Model training completed"
}

test_installation() {
    log_info "Testing the installation..."
    
    cd src
    
    # Test basic inference
    echo "Testing basic inference..."
    $PYTHON_CMD inference.py --input "YESTERDAY I GO STORE WITH FRIEND"
    
    # Test file processing if sample file exists
    if [ -f "../examples/sample_inputs.txt" ]; then
        echo "Testing file processing..."
        $PYTHON_CMD inference.py --input_file ../examples/sample_inputs.txt --output_file ../results/test_output.txt
    fi
    
    cd ..
    log_success "Installation test passed"
}

run_demo() {
    log_info "Running demonstration..."
    
    if [ -f "examples/demo.py" ]; then
        cd examples
        $PYTHON_CMD demo.py
        cd ..
        log_success "Demo completed"
    else
        log_warning "Demo script not found, skipping demo"
    fi
}

setup_jupyter() {
    log_info "Setting up Jupyter notebook..."
    
    # Install Jupyter kernel for the virtual environment
    pip install ipykernel
    $PYTHON_CMD -m ipykernel install --user --name=$VENV_NAME --display-name="ASL Translation"
    
    log_success "Jupyter kernel installed"
    log_info "You can now run: jupyter notebook notebooks/demo.ipynb"
}

create_usage_scripts() {
    log_info "Creating usage scripts..."
    
    # Create activation script
    cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the ASL Translation environment

source asl_env/bin/activate
echo "🚀 ASL Translation environment activated"
echo "Usage examples:"
echo "  cd src && python inference.py --interactive"
echo "  cd examples && python demo.py"
echo "  jupyter notebook notebooks/demo.ipynb"
EOF
    
    chmod +x activate.sh
    
    # Create quick test script  
    cat > quick_test.sh << 'EOF'
#!/bin/bash
# Quick test of the ASL translation system

source asl_env/bin/activate
cd src
echo "Testing ASL translation..."
python inference.py --input "YESTERDAY I GO STORE WITH FRIEND"
python inference.py --input "MORNING COFFEE I DRINK HOT"
echo "✅ Quick test completed!"
EOF
    
    chmod +x quick_test.sh
    
    log_success "Usage scripts created"
}

print_final_instructions() {
    echo ""
    echo "🎉 Installation completed successfully!"
    echo "====================================="
    echo ""
    echo "🚀 Quick Start:"
    echo "   ./activate.sh                    # Activate environment"
    echo "   ./quick_test.sh                  # Run quick test"
    echo ""  
    echo "📖 Usage Examples:"
    echo "   source asl_env/bin/activate      # Activate environment first"
    echo "   cd src"
    echo "   python inference.py --interactive                    # Interactive mode"
    echo "   python inference.py --input 'YESTERDAY I GO STORE'   # Single translation"
    echo "   python inference.py --input_file ../examples/sample_inputs.txt  # File processing"
    echo ""
    echo "🎭 Demos:"
    echo "   cd examples && python demo.py                        # Full demo"
    echo "   jupyter notebook notebooks/demo.ipynb               # Interactive notebook"
    echo ""
    echo "⚡ Model Quantization (for mobile):"
    echo "   cd src"
    echo "   python quantize.py --model_path ../models/distilt5-asl-finetuned --output_path ../models/quantized"
    echo ""
    echo "📁 Project Structure:"
    echo "   data/            - Training datasets"
    echo "   src/             - Source code (train.py, inference.py, etc.)"
    echo "   models/          - Trained models"
    echo "   examples/        - Demo scripts and sample files"
    echo "   notebooks/       - Jupyter notebooks"
    echo "   results/         - Output files"
    echo ""
    echo "🔧 Environment:"
    echo "   Virtual Environment: $VENV_NAME"
    echo "   Python Version: $($PYTHON_CMD --version)"
    echo "   Model: T5-efficient-tiny (~67MB)"
    echo ""
    echo "📞 Support:"
    echo "   - Check README.md for detailed documentation"
    echo "   - Check the source files for implementation details"
    echo "   - Open GitHub issues for bugs/questions"
    echo ""
}

cleanup_on_error() {
    log_error "Installation failed. Cleaning up..."
    
    # Deactivate virtual environment if active
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate 2>/dev/null || true
    fi
    
    # Optionally remove virtual environment
    read -p "Remove virtual environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $VENV_NAME
        log_info "Removed virtual environment"
    fi
}

# Main installation process
main() {
    print_banner
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Parse command line arguments
    SKIP_TRAINING=false
    SKIP_DEMO=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --skip-demo)
                SKIP_DEMO=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-training    Skip model training (faster setup)"
                echo "  --skip-demo       Skip running demo"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    check_requirements
    setup_environment
    install_dependencies
    
    # Generate data and train model (unless skipped)
    if [ "$SKIP_TRAINING" = false ]; then
        generate_training_data
        train_model
        test_installation
    else
        log_warning "Skipping training - you'll need to train the model manually"
    fi
    
    # Setup additional components
    setup_jupyter
    create_usage_scripts
    
    # Run demo (unless skipped)
    if [ "$SKIP_DEMO" = false ] && [ "$SKIP_TRAINING" = false ]; then
        run_demo
    fi
    
    print_final_instructions
}

# Run main installation
main "$@"