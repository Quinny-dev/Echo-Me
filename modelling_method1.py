#!/usr/bin/env python3
"""
Improved Sign Language Recognition Model
Training on First 10 Words from 1000+ Word Dataset

This script trains an improved model using better preprocessing, 
data augmentation, and architecture.
"""

import os
import numpy as np
import json
import time
from collections import Counter

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Masking, 
                                   Bidirectional, Input, BatchNormalization,
                                   Conv1D, MaxPooling1D, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

# Configuration
DATA_PATH = "Data"
NUM_FEATURES = 1662
WINDOW_SIZE = 100
STEP_SIZE = 20  # More overlap for better learning
MIN_SEQUENCE_LENGTH = 30  # Reduced to use more files
MAX_WORDS = 10  # Testing on first 10 words

print(f"\nConfiguration:")
print(f"   - Data path: {DATA_PATH}")
print(f"   - Features per frame: {NUM_FEATURES}")
print(f"   - Window size: {WINDOW_SIZE}")
print(f"   - Step size: {STEP_SIZE}")
print(f"   - Min sequence length: {MIN_SEQUENCE_LENGTH}")
print(f"   - Training on first {MAX_WORDS} words")

def normalize_keypoints(seq):
    """Enhanced normalization with stability checks"""
    seq = np.array(seq, dtype=np.float32)
    if seq.shape[0] == 0:
        return seq
    
    # Remove any NaN or inf values
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize pose landmarks (first 132 features: 33 landmarks * 4)
    if seq.shape[1] >= 132:
        pose_landmarks = seq[:, :132].copy()
        # Use center of pose as reference (average of first few pose points)
        center_points = pose_landmarks[:, :8].reshape(-1, 2, 4)[:, :, :2]  # First 2 pose points x,y
        if center_points.size > 0:
            ref_center = np.mean(center_points[0], axis=0)  # x, y reference
            # Normalize all pose x,y coordinates
            for i in range(33):  # 33 pose landmarks
                seq[:, i*4:i*4+2] -= ref_center
    
    # Robust z-score normalization
    mean_vals = np.mean(seq, axis=0, keepdims=True)
    std_vals = np.std(seq, axis=0, keepdims=True)
    std_vals = np.where(std_vals < 1e-8, 1.0, std_vals)  # Avoid division by zero
    seq = (seq - mean_vals) / std_vals
    
    return seq

def augment_sequence(seq, max_augmentations=2):
    """Data augmentation with variety - conservative for short sequences"""
    augmented = [seq]  # Original
    
    # Only augment sequences that are long enough
    if len(seq) > MIN_SEQUENCE_LENGTH * 1.5:  # At least 45 frames for augmentation
        # Noise augmentation
        noise = np.random.normal(0, 0.015, seq.shape)  # Reduced noise level
        noisy_seq = seq + noise
        augmented.append(noisy_seq)
        
        # Time shift (only for longer sequences)
        if len(seq) > 80 and max_augmentations > 1:
            # Slight compression (remove every 10th frame)
            indices = [i for i in range(len(seq)) if i % 10 != 0]
            if len(indices) >= MIN_SEQUENCE_LENGTH:
                compressed = seq[indices]
                augmented.append(compressed)
    
    return augmented[:max_augmentations + 1]  # Limit total augmentations

def load_and_preprocess_data():
    """Load data for first 10 words with enhanced preprocessing - using ALL files"""
    
    all_labels = sorted(os.listdir(DATA_PATH))[:MAX_WORDS]  # First 10 words
    print(f"Training on first {len(all_labels)} words: {all_labels}")
    
    label_map = {label: i for i, label in enumerate(all_labels)}
    
    X, y, video_sources = [], [], []
    
    total_videos_contributing = 0
    total_files_found = 0
    total_files_used = 0
    
    for label in all_labels:
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist!")
            continue
            
        video_count = 0
        label_samples = 0
        
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        total_files_found += len(files)
        print(f"Processing {len(files)} files for '{label}'...")
        
        for file in files:
            try:
                seq = np.load(os.path.join(folder, file))
                
                # Use ALL files - adjust window size for very short sequences
                if len(seq) < 10:  # Only skip extremely short sequences
                    print(f"   Skipping {file}: too short ({len(seq)} < 10 frames)")
                    continue
                
                total_files_used += 1
                
                if len(seq) < MIN_SEQUENCE_LENGTH:
                    print(f"   Short sequence {file}: {len(seq)} frames - using as single sample")
                    # For very short sequences, use them as single samples with padding
                    padded = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
                    seq_normalized = normalize_keypoints(seq)
                    padded[:len(seq_normalized)] = seq_normalized
                    X.append(padded)
                    y.append(label_map[label])
                    video_sources.append(f"{label}_{file}_short_padded")
                    label_samples += 1
                    video_count += 1
                    continue
                
                seq = normalize_keypoints(seq)
                
                # Apply limited augmentation for longer sequences
                if len(seq) >= WINDOW_SIZE:
                    augmented_seqs = augment_sequence(seq, max_augmentations=1)
                else:
                    augmented_seqs = [seq]  # No augmentation for short sequences
                
                file_contributed = False
                for aug_idx, aug_seq in enumerate(augmented_seqs):
                    # Create sliding windows or pad as needed
                    if len(aug_seq) >= WINDOW_SIZE:
                        # Use sliding windows for long sequences
                        for start in range(0, len(aug_seq) - WINDOW_SIZE + 1, STEP_SIZE):
                            window = aug_seq[start:start+WINDOW_SIZE]
                            X.append(window)
                            y.append(label_map[label])
                            video_sources.append(f"{label}_{file}_{start}_{aug_idx}")
                            label_samples += 1
                        file_contributed = True
                    else:
                        # Pad shorter sequences to WINDOW_SIZE
                        padded = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
                        padded[:len(aug_seq)] = aug_seq
                        X.append(padded)
                        y.append(label_map[label])
                        video_sources.append(f"{label}_{file}_padded_{aug_idx}")
                        label_samples += 1
                        file_contributed = True
                
                if file_contributed:
                    video_count += 1
                
            except Exception as e:
                print(f"   Error processing {file}: {e}")
                continue
        
        print(f"   {label}: {video_count} videos contributed {label_samples} samples")
        total_videos_contributing += video_count
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"\nDataset Summary:")
    print(f"   - Total .npy files found: {total_files_found}")
    print(f"   - Files successfully used: {total_files_used}")
    print(f"   - Files skipped: {total_files_found - total_files_used}")
    print(f"   - Videos contributing samples: {total_videos_contributing}")
    print(f"   - Total samples created: {len(X)}")
    print(f"   - Input shape: {X.shape}")
    print(f"   - Class distribution: {dict(Counter(y))}")
    
    # Show per-class breakdown
    print(f"\nPer-class sample count:")
    for i, label in enumerate(all_labels):
        count = sum(1 for y_val in y if y_val == i)
        print(f"   - {label}: {count} samples")
    
    return X, y, all_labels, video_sources

def visualize_class_distribution(y, all_labels):
    """Visualize class distribution"""
    plt.figure(figsize=(12, 6))
    class_counts = Counter(y)
    labels = [all_labels[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]

    plt.bar(labels, counts, color='skyblue', edgecolor='navy')
    plt.title('Class Distribution - Sample Count per Sign', fontsize=14, fontweight='bold')
    plt.xlabel('Sign Labels')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    for i, count in enumerate(counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📈 Class balance analysis:")
    min_samples = min(counts)
    max_samples = max(counts)
    print(f"   - Min samples per class: {min_samples}")
    print(f"   - Max samples per class: {max_samples}")
    print(f"   - Imbalance ratio: {max_samples/min_samples:.2f}")

def create_video_based_split(video_sources, all_labels, test_size=0.2, random_state=42):
    """Create train-test split based on videos to avoid data leakage"""
    
    # Extract unique video identifiers (remove augmentation and window info)
    unique_videos = set()
    video_to_indices = {}
    
    for i, src in enumerate(video_sources):
        # Extract base video name (label_filename)
        parts = src.split('_')
        video_key = '_'.join(parts[:2])  # label_filename
        
        unique_videos.add(video_key)
        
        if video_key not in video_to_indices:
            video_to_indices[video_key] = []
        video_to_indices[video_key].append(i)
    
    print(f"\n🎬 Found {len(unique_videos)} unique videos")
    
    # Split videos by label to maintain class distribution
    train_indices = []
    test_indices = []
    small_class_videos = []  # Track classes with very few videos
    
    for label in all_labels:
        label_videos = [v for v in unique_videos if v.startswith(label + '_')]
        
        if len(label_videos) <= 1:
            # For classes with 1 or 0 videos, put all in training
            print(f"   {label}: {len(label_videos)} videos (too few for split - adding to train)")
            for vid in label_videos:
                train_indices.extend(video_to_indices[vid])
            small_class_videos.extend(label_videos)
        elif len(label_videos) == 2:
            # For classes with exactly 2 videos, put 1 in each set
            train_vids = [label_videos[0]]
            test_vids = [label_videos[1]]
            
            for vid in train_vids:
                train_indices.extend(video_to_indices[vid])
            for vid in test_vids:
                test_indices.extend(video_to_indices[vid])
            
            print(f"   {label}: 1 train video, 1 test video (minimal split)")
        else:
            # For classes with 3+ videos, use normal split
            # Ensure at least 1 video in each set
            min_test_videos = max(1, int(len(label_videos) * test_size))
            min_test_videos = min(min_test_videos, len(label_videos) - 1)  # Leave at least 1 for train
            
            train_vids, test_vids = train_test_split(
                label_videos, 
                test_size=min_test_videos, 
                random_state=random_state
            )
            
            # Add corresponding indices
            for vid in train_vids:
                train_indices.extend(video_to_indices[vid])
            for vid in test_vids:
                test_indices.extend(video_to_indices[vid])
            
            print(f"   {label}: {len(train_vids)} train videos, {len(test_vids)} test videos")
    
    # Check if we have any test samples
    if len(test_indices) == 0:
        print("   ⚠️  Warning: No test samples available. Using 20% of training data as test set.")
        # Fallback: use sample-based split but try to avoid data leakage
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # Get labels for all samples
        sample_labels = []
        for src in video_sources:
            label = src.split('_')[0]
            sample_labels.append(all_labels.index(label))
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(sss.split(range(len(video_sources)), sample_labels))
        return list(train_idx), list(test_idx)
    
    print(f"\n📊 Split Results:")
    print(f"   - Total train samples: {len(train_indices)}")
    print(f"   - Total test samples: {len(test_indices)}")
    if small_class_videos:
        print(f"   - Classes with insufficient videos for splitting: {len(small_class_videos)} videos")
    
    return train_indices, test_indices

def create_improved_model(input_shape, num_classes, model_type="hybrid", dataset_size="small"):
    """Create improved model with optimal architecture for your dataset size"""
    
    # Adjust model complexity based on dataset size
    if dataset_size == "small":  # < 200 samples
        if model_type == "simple_lstm":
            model = Sequential([
                Input(shape=input_shape),
                Masking(mask_value=0.0),
                LSTM(32, return_sequences=True, dropout=0.4, recurrent_dropout=0.3),
                LSTM(16, dropout=0.4, recurrent_dropout=0.3),
                Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
                Dropout(0.6),
                Dense(num_classes, activation='softmax')
            ])
        
        elif model_type == "cnn_lstm":
            model = Sequential([
                Input(shape=input_shape),
                Conv1D(16, 5, activation='relu', padding='same'),
                BatchNormalization(),
                Conv1D(32, 5, activation='relu', padding='same'),
                MaxPooling1D(2),
                Dropout(0.4),
                LSTM(32, return_sequences=True, dropout=0.4),
                LSTM(16, dropout=0.4),
                Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
                Dropout(0.6),
                Dense(num_classes, activation='softmax')
            ])
        
        else:  # hybrid (recommended for small datasets)
            model = Sequential([
                Input(shape=input_shape),
                Masking(mask_value=0.0),
                
                # Simplified feature extraction
                Conv1D(32, 5, activation='relu', padding='same'),
                BatchNormalization(),
                Dropout(0.3),
                
                Conv1D(64, 5, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.4),
                
                # Simpler temporal modeling
                LSTM(32, return_sequences=True, dropout=0.4),
                LSTM(16, dropout=0.4),
                
                # Classification head
                BatchNormalization(),
                Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
                Dropout(0.6),
                Dense(num_classes, activation='softmax')
            ])
            
    else:  # Original architecture for larger datasets
        if model_type == "simple_lstm":
            model = Sequential([
                Input(shape=input_shape),
                Masking(mask_value=0.0),
                LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
                LSTM(32, dropout=0.3, recurrent_dropout=0.2),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
        
        elif model_type == "cnn_lstm":
            model = Sequential([
                Input(shape=input_shape),
                Conv1D(32, 5, activation='relu', padding='same'),
                BatchNormalization(),
                Conv1D(64, 5, activation='relu', padding='same'),
                MaxPooling1D(2),
                Dropout(0.3),
                LSTM(64, return_sequences=True, dropout=0.3),
                LSTM(32, dropout=0.3),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
        
        else:  # hybrid (recommended)
            model = Sequential([
                Input(shape=input_shape),
                Masking(mask_value=0.0),
                
                # Feature extraction with Conv1D
                Conv1D(64, 3, activation='relu', padding='same'),
                BatchNormalization(),
                Dropout(0.2),
                
                Conv1D(128, 3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.3),
                
                # Temporal modeling with LSTM
                Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
                Bidirectional(LSTM(32, dropout=0.3)),
                
                # Classification head
                BatchNormalization(),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
    
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Progress', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Progress', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print best scores
    best_val_acc = max(history.history['val_accuracy'])
    best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n📈 Training Results:")
    print(f"   - Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_acc_epoch})")
    print(f"   - Final validation accuracy: {final_val_acc:.4f}")
    print(f"   - Total epochs trained: {len(history.history['accuracy'])}")

def analyze_predictions(model, X_test, y_test_cat, all_labels):
    """Analyze model predictions and create confusion matrix"""
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_cat, axis=1)
    
    # Get unique classes that actually appear in test set
    unique_test_classes = np.unique(np.concatenate([true_classes, predicted_classes]))
    test_labels = [all_labels[i] for i in sorted(unique_test_classes)]
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(f"   (Note: Only showing classes present in test set)")
    print(classification_report(
        true_classes, predicted_classes, 
        labels=sorted(unique_test_classes),
        target_names=test_labels,
        zero_division=0
    ))
    
    # Full confusion matrix with all classes
    cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(all_labels)))
    
    # Only show non-empty classes in visualization
    non_empty_classes = []
    non_empty_indices = []
    for i in range(len(all_labels)):
        if i in unique_test_classes:
            non_empty_classes.append(all_labels[i])
            non_empty_indices.append(i)
    
    if len(non_empty_indices) > 1:  # Only create plot if we have multiple classes
        cm_subset = cm[np.ix_(non_empty_indices, non_empty_indices)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=non_empty_classes, yticklabels=non_empty_classes)
        plt.title('Confusion Matrix - Sign Language Recognition\n(Classes present in test set)', 
                  fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Per-class accuracy analysis
    print(f"\n🎯 Per-class Performance Analysis:")
    print(f"   (Classes in test set)")
    
    for i in sorted(unique_test_classes):
        class_mask = (true_classes == i)
        if np.sum(class_mask) > 0:  # Only if class appears in test set
            class_correct = np.sum((predicted_classes == i) & (true_classes == i))
            class_total = np.sum(class_mask)
            class_accuracy = class_correct / class_total
            print(f"   - {all_labels[i]}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) "
                  f"[{class_correct}/{class_total} correct]")
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   - Classes in training: {len(all_labels)}")
    print(f"   - Classes in test set: {len(unique_test_classes)}")
    print(f"   - Missing from test: {set(range(len(all_labels))) - set(unique_test_classes)}")
    if len(set(range(len(all_labels))) - set(unique_test_classes)) > 0:
        missing_labels = [all_labels[i] for i in set(range(len(all_labels))) - set(unique_test_classes)]
        print(f"   - Missing labels: {missing_labels}")

def save_model_and_config(model, all_labels, test_accuracy, X_train, X_test):
    """Save model, configuration, and label mapping"""
    # Save the final model
    model.save('sign_language_model_final.h5')
    print("✅ Model saved as 'sign_language_model_final.h5'")

    # Save label mapping
    label_mapping = {label: i for i, label in enumerate(all_labels)}
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print("✅ Label mapping saved as 'label_mapping.json'")

    # Save training configuration
    config = {
        'num_features': NUM_FEATURES,
        'window_size': WINDOW_SIZE,
        'labels': all_labels,
        'model_type': 'hybrid',
        'final_test_accuracy': float(test_accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✅ Model configuration saved as 'model_config.json'")

    print(f"\n🎉 Model files ready for real-time inference!")
    print(f"   Final accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

def test_sample_predictions(model, X_test, y_test_cat, all_labels):
    """Test a few predictions to ensure model is working"""
    print("\n🧪 Testing model predictions on sample data:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        sample_input = X_test[idx:idx+1]  # Single sample
        prediction = model.predict(sample_input, verbose=0)[0]
        
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        true_class = np.argmax(y_test_cat[idx])
        
        print(f"   Sample {i+1}:")
        print(f"     True: {all_labels[true_class]}")
        print(f"     Predicted: {all_labels[predicted_class]} (confidence: {confidence:.3f})")
        print(f"     Correct: {'✅' if predicted_class == true_class else '❌'}")

def visualize_class_distribution(y, all_labels):
    """Visualize class distribution"""
    plt.figure(figsize=(12, 6))
    class_counts = Counter(y)
    labels = [all_labels[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]

    plt.bar(labels, counts, color='skyblue', edgecolor='navy')
    plt.title('Class Distribution - Sample Count per Sign', fontsize=14, fontweight='bold')
    plt.xlabel('Sign Labels')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    for i, count in enumerate(counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nClass balance analysis:")
    min_samples = min(counts)
    max_samples = max(counts)
    print(f"   - Min samples per class: {min_samples}")
    print(f"   - Max samples per class: {max_samples}")
    print(f"   - Imbalance ratio: {max_samples/min_samples:.2f}")

def create_video_based_split(video_sources, all_labels, test_size=0.2, random_state=42):
    """Create train-test split based on videos to avoid data leakage"""
    
    # Extract unique video identifiers (remove augmentation and window info)
    unique_videos = set()
    video_to_indices = {}
    
    for i, src in enumerate(video_sources):
        # Extract base video name (label_filename)
        parts = src.split('_')
        video_key = '_'.join(parts[:2])  # label_filename
        
        unique_videos.add(video_key)
        
        if video_key not in video_to_indices:
            video_to_indices[video_key] = []
        video_to_indices[video_key].append(i)
    
    print(f"\nFound {len(unique_videos)} unique videos")
    
    # Split videos by label to maintain class distribution
    train_indices = []
    test_indices = []
    small_class_videos = []  # Track classes with very few videos
    
    for label in all_labels:
        label_videos = [v for v in unique_videos if v.startswith(label + '_')]
        
        if len(label_videos) <= 1:
            # For classes with 1 or 0 videos, put all in training
            print(f"   {label}: {len(label_videos)} videos (too few for split - adding to train)")
            for vid in label_videos:
                train_indices.extend(video_to_indices[vid])
            small_class_videos.extend(label_videos)
        elif len(label_videos) == 2:
            # For classes with exactly 2 videos, put 1 in each set
            train_vids = [label_videos[0]]
            test_vids = [label_videos[1]]
            
            for vid in train_vids:
                train_indices.extend(video_to_indices[vid])
            for vid in test_vids:
                test_indices.extend(video_to_indices[vid])
            
            print(f"   {label}: 1 train video, 1 test video (minimal split)")
        else:
            # For classes with 3+ videos, use normal split
            # Ensure at least 1 video in each set
            min_test_videos = max(1, int(len(label_videos) * test_size))
            min_test_videos = min(min_test_videos, len(label_videos) - 1)  # Leave at least 1 for train
            
            train_vids, test_vids = train_test_split(
                label_videos, 
                test_size=min_test_videos, 
                random_state=random_state
            )
            
            # Add corresponding indices
            for vid in train_vids:
                train_indices.extend(video_to_indices[vid])
            for vid in test_vids:
                test_indices.extend(video_to_indices[vid])
            
            print(f"   {label}: {len(train_vids)} train videos, {len(test_vids)} test videos")
    
    # Check if we have any test samples
    if len(test_indices) == 0:
        print("   Warning: No test samples available. Using 20% of training data as test set.")
        # Fallback: use sample-based split but try to avoid data leakage
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        
        # Get labels for all samples
        sample_labels = []
        for src in video_sources:
            label = src.split('_')[0]
            sample_labels.append(all_labels.index(label))
        
        train_idx, test_idx = next(sss.split(range(len(video_sources)), sample_labels))
        return list(train_idx), list(test_idx)
    
    print(f"\nSplit Results:")
    print(f"   - Total train samples: {len(train_indices)}")
    print(f"   - Total test samples: {len(test_indices)}")
    if small_class_videos:
        print(f"   - Classes with insufficient videos for splitting: {len(small_class_videos)} videos")
    
    return train_indices, test_indices

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Progress', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Progress', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print best scores
    best_val_acc = max(history.history['val_accuracy'])
    best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\nTraining Results:")
    print(f"   - Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_acc_epoch})")
    print(f"   - Final validation accuracy: {final_val_acc:.4f}")
    print(f"   - Total epochs trained: {len(history.history['accuracy'])}")

def analyze_predictions(model, X_test, y_test_cat, all_labels):
    """Analyze model predictions and create confusion matrix"""
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_cat, axis=1)
    
    # Get unique classes that actually appear in test set
    unique_test_classes = np.unique(np.concatenate([true_classes, predicted_classes]))
    test_labels = [all_labels[i] for i in sorted(unique_test_classes)]
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(f"   (Note: Only showing classes present in test set)")
    print(classification_report(
        true_classes, predicted_classes, 
        labels=sorted(unique_test_classes),
        target_names=test_labels,
        zero_division=0
    ))
    
    # Full confusion matrix with all classes
    cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(all_labels)))
    
    # Only show non-empty classes in visualization
    non_empty_classes = []
    non_empty_indices = []
    for i in range(len(all_labels)):
        if i in unique_test_classes:
            non_empty_classes.append(all_labels[i])
            non_empty_indices.append(i)
    
    if len(non_empty_indices) > 1:  # Only create plot if we have multiple classes
        cm_subset = cm[np.ix_(non_empty_indices, non_empty_indices)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=non_empty_classes, yticklabels=non_empty_classes)
        plt.title('Confusion Matrix - Sign Language Recognition\n(Classes present in test set)', 
                  fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Per-class accuracy analysis
    print(f"\nPer-class Performance Analysis:")
    print(f"   (Classes in test set)")
    
    for i in sorted(unique_test_classes):
        class_mask = (true_classes == i)
        if np.sum(class_mask) > 0:  # Only if class appears in test set
            class_correct = np.sum((predicted_classes == i) & (true_classes == i))
            class_total = np.sum(class_mask)
            class_accuracy = class_correct / class_total
            print(f"   - {all_labels[i]}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) "
                  f"[{class_correct}/{class_total} correct]")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"   - Classes in training: {len(all_labels)}")
    print(f"   - Classes in test set: {len(unique_test_classes)}")
    print(f"   - Missing from test: {set(range(len(all_labels))) - set(unique_test_classes)}")
    if len(set(range(len(all_labels))) - set(unique_test_classes)) > 0:
        missing_labels = [all_labels[i] for i in set(range(len(all_labels))) - set(unique_test_classes)]
        print(f"   - Missing labels: {missing_labels}")

def save_model_and_config(model, all_labels, test_accuracy, X_train, X_test):
    """Save model, configuration, and label mapping"""
    # Save the final model
    model.save('sign_language_model_final.h5')
    print("Model saved as 'sign_language_model_final.h5'")

    # Save label mapping
    label_mapping = {label: i for i, label in enumerate(all_labels)}
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print("Label mapping saved as 'label_mapping.json'")

    # Save training configuration
    config = {
        'num_features': NUM_FEATURES,
        'window_size': WINDOW_SIZE,
        'labels': all_labels,
        'model_type': 'hybrid',
        'final_test_accuracy': float(test_accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Model configuration saved as 'model_config.json'")

    print(f"\nModel files ready for real-time inference!")
    print(f"   Final accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

def test_sample_predictions(model, X_test, y_test_cat, all_labels):
    """Test a few predictions to ensure model is working"""
    print("\nTesting model predictions on sample data:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        sample_input = X_test[idx:idx+1]  # Single sample
        prediction = model.predict(sample_input, verbose=0)[0]
        
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        true_class = np.argmax(y_test_cat[idx])
        
        print(f"   Sample {i+1}:")
        print(f"     True: {all_labels[true_class]}")
        print(f"     Predicted: {all_labels[predicted_class]} (confidence: {confidence:.3f})")
        print(f"     Correct: {'Yes' if predicted_class == true_class else 'No'}")

def main():
    """Main training function"""
    print("Starting Sign Language Model Training\n")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    start_time = time.time()
    X, y, all_labels, video_sources = load_and_preprocess_data()
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    
    # Visualize data distribution
    visualize_class_distribution(y, all_labels)
    
    # Create video-based split
    print("\nCreating train-test split (avoiding data leakage)...")
    train_indices, test_indices = create_video_based_split(video_sources, all_labels)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"\nSplit Summary:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Testing samples: {len(X_test)}")
    print(f"   - Train class distribution: {dict(Counter(y_train))}")
    print(f"   - Test class distribution: {dict(Counter(y_test))}")
    
    # Convert to categorical and compute class weights
    y_train_cat = to_categorical(y_train, len(all_labels))
    y_test_cat = to_categorical(y_test, len(all_labels))
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Determine dataset size category
    dataset_size = "small" if len(X) < 200 else "medium" if len(X) < 1000 else "large"
    print(f"Dataset size category: {dataset_size} ({len(X)} samples)")
    
    if dataset_size == "small":
        print("Small dataset detected - using simplified model and adjusted parameters")
    
    # Create model
    print(f"\nCreating model architecture...")
    model = create_improved_model((WINDOW_SIZE, NUM_FEATURES), len(all_labels), "hybrid", dataset_size)
    
    # Compile with appropriate optimizer (lower learning rate for small datasets)
    learning_rate = 0.0005 if dataset_size == "small" else 0.001
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    
    # Setup callbacks (adjusted for dataset size)
    patience = 10 if dataset_size == "small" else 15
    lr_patience = 5 if dataset_size == "small" else 7
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_sign_language_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training parameters (adjusted for dataset size)
    EPOCHS = 200 if dataset_size == "small" else 100
    BATCH_SIZE = min(16 if dataset_size == "small" else 32, len(X_train) // 4)  # Smaller batches for small datasets
    BATCH_SIZE = max(1, BATCH_SIZE)  # Ensure at least batch size 1
    
    print(f"\nTraining Configuration:")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Callbacks: Early Stopping, Learning Rate Reduction, Model Checkpoint")
    print(f"   - Class weights: Applied")
    
    # Train the model
    print(f"\nStarting training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nFinal Test Results:")
    print(f"   - Test Loss: {test_loss:.4f}")
    print(f"   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    
    # Analyze predictions
    analyze_predictions(model, X_test, y_test_cat, all_labels)
    
    # Save model and configuration
    save_model_and_config(model, all_labels, test_accuracy, X_train, X_test)
    
    # Test sample predictions
    test_sample_predictions(model, X_test, y_test_cat, all_labels)
    
    print(f"\nTraining completed successfully!")
    print(f"Ready for real-time inference!")

if __name__ == "__main__":
    main()(model, X_test, y_test_cat, all_labels)
    
    # Save model and configuration
    save_model_and_config(model, all_labels, test_accuracy, X_train, X_test)
    
    # Test sample predictions
    test_sample_predictions(model, X_test, y_test_cat, all_labels)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"🚀 Ready for real-time inference!")

if __name__ == "__main__":
    main()