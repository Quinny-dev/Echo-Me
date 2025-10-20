#!/usr/bin/env python3
"""
Setup and installation script for ASL Gloss to English Translation System.
Automates the entire setup process from dataset generation to model training.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description="", cwd=None):
    """Run shell command with error handling."""
    logger.info(f"🔄 {description}")
    logger.info(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.info(result.stdout)
        logger.info(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        "torch", "transformers", "datasets", "accelerate", 
        "evaluate", "optimum", "pandas", "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("✅ All dependencies available")
    return True

def install_dependencies():
    """Install required dependencies."""
    logger.info("📦 Installing dependencies...")
    
    return run_command(
        "pip install -r requirements.txt",
        "Installing Python packages"
    )

def create_directory_structure():
    """Create necessary directories."""
    logger.info("📁 Creating directory structure...")
    
    directories = [
        "data",
        "src", 
        "models",
        "examples",
        "notebooks",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("✅ Directory structure created")
    return True

def generate_dataset():
    """Generate synthetic training dataset."""
    logger.info("🎲 Generating synthetic dataset...")
    
    return run_command(
        "python synthetic_dataset.py",
        "Generating ASL gloss training data",
        cwd="data"
    )

def train_model(args):
    """Train the translation model."""
    logger.info("🏋️ Training ASL translation model...")
    
    # Build training command
    cmd_parts = [
        "python train.py",
        f"--model_name {args.model_name}",
        f"--num_epochs {args.epochs}",
        f"--batch_size {args.batch_size}",
        f"--learning_rate {args.learning_rate}"
    ]
    
    if args.device != "auto":
        cmd_parts.append(f"--device {args.device}")
    
    command = " ".join(cmd_parts)
    
    return run_command(
        command,
        "Training translation model",
        cwd="src"
    )

def test_model():
    """Test the trained model."""
    logger.info("🧪 Testing trained model...")
    
    test_commands = [
        'python inference.py --input "YESTERDAY I GO STORE WITH FRIEND"',
        'python inference.py --input_file ../examples/sample_inputs.txt --output_file ../results/test_translations.txt'
    ]
    
    success = True
    for cmd in test_commands:
        if not run_command(cmd, "Testing model inference", cwd="src"):
            success = False
    
    return success

def run_demo():
    """Run the demo script."""
    logger.info("🎭 Running demo...")
    
    return run_command(
        "python demo.py",
        "Running translation demo",
        cwd="examples"
    )

def quantize_model(args):
    """Quantize model for mobile deployment."""
    if not args.quantize:
        return True
    
    logger.info("⚡ Quantizing model for mobile deployment...")
    
    return run_command(
        f"python quantize.py --model_path ../models/distilt5-asl-finetuned --output_path ../models/quantized --method both",
        "Quantizing model",
        cwd="src"
    )

def create_quick_start_guide():
    """Create a quick start guide."""
    logger.info("📝 Creating quick start guide...")
    
    guide_content = """# Quick Start Guide - ASL Gloss Translation

## 🚀 Ready to Use!

Your ASL Gloss to English translation system is now set up and ready to use.

### Basic Usage

1. **Single Translation**:
   ```bash
   cd src
   python inference.py --input "YESTERDAY I GO STORE WITH FRIEND"
   ```

2. **Interactive Mode**:
   ```bash
   cd src  
   python inference.py --interactive
   ```

3. **File Processing**:
   ```bash
   cd src
   python inference.py --input_file ../examples/sample_inputs.txt
   ```

### Demo and Testing

1. **Run Full Demo**:
   ```bash
   cd examples
   python demo.py
   ```

2. **Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/demo.ipynb
   ```

### Model Information

- **Base Model**: DistilT5 (lightweight, ~67MB)
- **Training Data**: Synthetic ASL gloss-English pairs
- **Performance**: ~100-200ms per sentence on CPU
- **Mobile Ready**: Use quantized versions in models/quantized/

### Next Steps

1. **Add Real Data**: Replace synthetic data with real ASL gloss pairs
2. **Fine-tune**: Retrain on domain-specific data
3. **Deploy**: Use quantized models for mobile/web deployment
4. **Integrate**: Embed into applications or services

### Troubleshooting

- **Model not found**: Make sure training completed successfully
- **CUDA errors**: Use `--device cpu` for CPU-only training
- **Memory issues**: Reduce batch size with `--batch_size 4`

### Support

- Check the README.md for detailed documentation
- View example outputs in results/ directory
- Open issues on GitHub for bugs or questions
"""
    
    with open("QUICKSTART.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    logger.info("✅ Quick start guide created")
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup ASL Gloss to English Translation System")
    
    # Setup options
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset generation (use existing)")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip model training (use existing model)")
    parser.add_argument("--skip-demo", action="store_true",
                       help="Skip running demo")
    parser.add_argument("--quantize", action="store_true",
                       help="Quantize model for mobile deployment")
    
    # Training parameters
    parser.add_argument("--model_name", type=str, default="google/t5-efficient-tiny",
                       help="Base model name")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device for training")
    
    return parser.parse_args()

def main():
    """Main setup function."""
    print("🚀 ASL Gloss to English Translation System Setup")
    print("=" * 60)
    
    args = parse_arguments()
    
    setup_steps = [
        ("Directory Structure", create_directory_structure, True),
        ("Dependencies", lambda: install_dependencies() if not args.skip_install else True, not args.skip_install),
        ("Dataset Generation", lambda: generate_dataset() if not args.skip_dataset else True, not args.skip_dataset),
        ("Model Training", lambda: train_model(args) if not args.skip_training else True, not args.skip_training),
        ("Model Testing", test_model, not args.skip_training),
        ("Model Quantization", lambda: quantize_model(args), args.quantize),
        ("Demo", lambda: run_demo() if not args.skip_demo else True, not args.skip_demo),
        ("Quick Start Guide", create_quick_start_guide, True)
    ]
    
    failed_steps = []
    
    for step_name, step_function, should_run in setup_steps:
        if not should_run:
            logger.info(f"⏭️ Skipping {step_name}")
            continue
            
        logger.info(f"\n{'='*20} {step_name} {'='*20}")
        
        try:
            success = step_function()
            if not success:
                failed_steps.append(step_name)
                logger.error(f"❌ {step_name} failed")
                
                # Ask user if they want to continue
                response = input(f"\n❓ Continue setup despite {step_name} failure? (y/n): ")
                if response.lower() not in ['y', 'yes']:
                    logger.info("Setup aborted by user")
                    sys.exit(1)
            else:
                logger.info(f"✅ {step_name} completed successfully")
                
        except KeyboardInterrupt:
            logger.info("\n\n⏹️ Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    # Final summary
    print("\n" + "="*60)
    print("📋 SETUP SUMMARY")
    print("="*60)
    
    if not failed_steps:
        print("🎉 Setup completed successfully!")
        print("\n✅ Your ASL Gloss Translation System is ready to use!")
        print("\n🚀 Quick Start:")
        print("   cd src")
        print('   python inference.py --input "YESTERDAY I GO STORE"')
        print("   python inference.py --interactive")
        print("\n📖 Check QUICKSTART.md for more details")
        
    else:
        print("⚠️ Setup completed with some failures:")
        for step in failed_steps:
            print(f"   ❌ {step}")
        print("\n💡 You may need to manually complete failed steps")
        print("📖 Check the logs above for error details")
    
    print("\n📁 Project Structure:")
    structure_items = [
        "├── data/           # Training datasets", 
        "├── src/            # Source code (train.py, inference.py)",
        "├── models/         # Trained models",
        "├── examples/       # Demo scripts and sample inputs",
        "├── notebooks/      # Jupyter notebook demo", 
        "├── results/        # Output files and logs",
        "└── QUICKSTART.md   # Quick start guide"
    ]
    
    for item in structure_items:
        print(f"   {item}")
    
    print(f"\n🔧 Setup Configuration:")
    config_items = [
        f"Model: {args.model_name}",
        f"Epochs: {args.epochs}",
        f"Batch Size: {args.batch_size}", 
        f"Learning Rate: {args.learning_rate}",
        f"Device: {args.device}",
        f"Quantization: {'Enabled' if args.quantize else 'Disabled'}"
    ]
    
    for item in config_items:
        print(f"   • {item}")

if __name__ == "__main__":
    main()