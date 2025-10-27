"""
Label Loading Diagnostic Script
Checks if labels are being loaded correctly from your data folder structure
"""
from pathlib import Path

def load_labels_current_method():
    """Current method used in your code"""
    # Initialize data directory and empty label list
    data_dir = Path("data")
    label_dirs = []
    folder_order = ["None", "holds_data", "nonholds_data"]

    # Iterate through folders in specified order
    for folder_name in folder_order:
        folder_path = data_dir / folder_name
        # Check if folder exists and is a directory
        if folder_path.exists() and folder_path.is_dir():
            if folder_name == "None":
                # Add "None" as a label directly
                label_dirs.append("None")
            else:
                # Add all subdirectory names as labels
                for f in sorted(folder_path.iterdir()):
                    if f.is_dir():
                        label_dirs.append(f.name)

    return label_dirs


def load_labels_correct_method():
    """Correct method that matches your actual data structure"""
    # Initialize data directory and empty label list
    data_dir = Path("data")
    labels = []
    
    # Step 1: Add "None" class (from None folder)
    none_folder = data_dir / "None"
    if none_folder.exists() and none_folder.is_dir():
        labels.append("None")
    
    # Step 2: Add all subdirectories from holds_data (alphabetically)
    holds_folder = data_dir / "holds_data"
    if holds_folder.exists() and holds_folder.is_dir():
        # Iterate through sorted subdirectories
        for subfolder in sorted(holds_folder.iterdir()):
            if subfolder.is_dir():
                labels.append(subfolder.name)
    
    # Step 3: Add all subdirectories from nonholds_data (alphabetically)
    nonholds_folder = data_dir / "nonholds_data"
    if nonholds_folder.exists() and nonholds_folder.is_dir():
        # Iterate through sorted subdirectories
        for subfolder in sorted(nonholds_folder.iterdir()):
            if subfolder.is_dir():
                labels.append(subfolder.name)
    
    return labels


def analyze_data_structure():
    """Analyze the actual data folder structure"""
    data_dir = Path("data")
    
    # Print header
    print("=" * 60)
    print("DATA STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Check None folder
    none_folder = data_dir / "None"
    print(f"\nNone folder:")
    print(f"   Exists: {none_folder.exists()}")
    if none_folder.exists():
        # Count .npy files in None folder
        files = list(none_folder.glob("*.npy"))
        print(f"   Files: {len(files)}")
    
    # Check holds_data folder
    holds_folder = data_dir / "holds_data"
    print(f"\nholds_data folder:")
    print(f"   Exists: {holds_folder.exists()}")
    if holds_folder.exists():
        # Get all subdirectories
        subfolders = [f for f in holds_folder.iterdir() if f.is_dir()]
        print(f"   Subfolders: {len(subfolders)}")
        # List each subfolder with file count
        for sf in sorted(subfolders):
            files = list(sf.glob("*.npy"))
            print(f"      - {sf.name}: {len(files)} files")
    
    # Check nonholds_data folder
    nonholds_folder = data_dir / "nonholds_data"
    print(f"\nnonholds_data folder:")
    print(f"   Exists: {nonholds_folder.exists()}")
    if nonholds_folder.exists():
        # Get all subdirectories
        subfolders = [f for f in nonholds_folder.iterdir() if f.is_dir()]
        print(f"   Subfolders: {len(subfolders)}")
        # List each subfolder with file count
        for sf in sorted(subfolders):
            files = list(sf.glob("*.npy"))
            print(f"      - {sf.name}: {len(files)} files")


def check_model_output_size():
    """Check what the model expects"""
    import tensorflow as tf
    import numpy as np
    
    # Print header
    print("\n" + "=" * 60)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 60)
    
    try:
        model_path = "models/model_fast"
        
        # Try loading with Keras
        try:
            # Load model using Keras
            model = tf.keras.models.load_model(model_path)
            output_shape = model.output_shape
            num_classes = output_shape[-1]
            print(f"\nModel loaded successfully")
            print(f"   Output shape: {output_shape}")
            print(f"   Number of classes: {num_classes}")
            
            # Test prediction shape with dummy input
            dummy_input = np.zeros((1, model.input_shape[1], 130), dtype=np.float32)
            pred = model(dummy_input)
            print(f"   Prediction shape: {pred.shape}")
            print(f"   Expected: (1, {num_classes})")
            
        except Exception as e:
            print(f"Keras load failed: {e}")
            
            # Try SavedModel format
            loaded = tf.saved_model.load(model_path)
            infer = loaded.signatures['serving_default']
            
            # Get output specification
            output_spec = list(infer.structured_output_signature[1].values())[0]
            num_classes = output_spec.shape[-1]
            print(f"\nSavedModel loaded")
            print(f"   Output shape: {output_spec.shape}")
            print(f"   Number of classes: {num_classes}")
            
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None
    
    return num_classes


def compare_methods():
    """Compare both label loading methods"""
    # Print header
    print("\n" + "=" * 60)
    print("LABEL LOADING COMPARISON")
    print("=" * 60)
    
    # Load labels using both methods
    current = load_labels_current_method()
    correct = load_labels_correct_method()
    
    # Display labels from current method
    print(f"\nCurrent method loaded {len(current)} labels:")
    for i, label in enumerate(current):
        print(f"   {i:2d}: {label}")
    
    # Display labels from correct method
    print(f"\nCorrect method loaded {len(correct)} labels:")
    for i, label in enumerate(correct):
        print(f"   {i:2d}: {label}")
    
    # Check differences
    if current == correct:
        print("\nBoth methods produce IDENTICAL results!")
    else:
        print("\nMethods produce DIFFERENT results!")
        
        # Find differences
        only_in_current = set(current) - set(correct)
        only_in_correct = set(correct) - set(current)
        
        # Show labels unique to each method
        if only_in_current:
            print(f"\n   Labels only in current method: {only_in_current}")
        if only_in_correct:
            print(f"   Labels only in correct method: {only_in_correct}")
        
        # Check order differences
        if set(current) == set(correct):
            print("\n   Same labels, but DIFFERENT ORDER!")
            print("\n   Order matters for model predictions!")
    
    return current, correct


def main():
    """Run all diagnostics"""
    print("\nECHO ME - LABEL LOADING DIAGNOSTIC\n")
    
    # Analyze folder structure
    analyze_data_structure()
    
    # Compare label loading methods
    current_labels, correct_labels = compare_methods()
    
    # Check model expectations
    num_classes = check_model_output_size()
    
    # Final verification
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    
    if num_classes:
        # Display counts
        print(f"\n   Model expects: {num_classes} classes")
        print(f"   Current method loads: {len(current_labels)} labels")
        print(f"   Correct method loads: {len(correct_labels)} labels")
        
        # Check current method
        if len(current_labels) == num_classes:
            print("\n   Current method: Label count MATCHES model!")
        else:
            print(f"\n   Current method: Label count MISMATCH!")
            print(f"      Difference: {len(current_labels) - num_classes}")
        
        # Check correct method
        if len(correct_labels) == num_classes:
            print("\n   Correct method: Label count MATCHES model!")
        else:
            print(f"\n   Correct method: Label count MISMATCH!")
            print(f"      Difference: {len(correct_labels) - num_classes}")
    
    # Print recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if current_labels == correct_labels and num_classes and len(current_labels) == num_classes:
        print("\n   Everything looks good! Current method is working correctly.")
    else:
        print("\n   Issues detected. Review the analysis above.")
        print("\n   The label loading order MUST match the order used during training.")
        print("   Check your training script to see how labels were sorted.")


if __name__ == "__main__":
    main()