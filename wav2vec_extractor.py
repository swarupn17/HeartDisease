"""
wav2vec 2.0 Feature Extractor for Heart Sound Classification

This module provides feature extraction using pretrained wav2vec 2.0 model.
It converts raw audio into semantic embeddings without manual feature engineering.
"""

import numpy as np
import librosa
import torch
from transformers import AutoModel, AutoProcessor
import warnings

warnings.filterwarnings("ignore")

# Global model cache to avoid reloading
_wav2vec_model = None
_wav2vec_processor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_wav2vec_model(model_name="facebook/wav2vec2-base"):
    """
    Load pretrained wav2vec 2.0 model and processor.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tuple of (model, processor)
    """
    global _wav2vec_model, _wav2vec_processor
    
    if _wav2vec_model is None:
        print(f"📥 Loading wav2vec 2.0 model: {model_name}")
        try:
            _wav2vec_processor = AutoProcessor.from_pretrained(model_name)
            _wav2vec_model = AutoModel.from_pretrained(model_name).to(DEVICE)
            _wav2vec_model.eval()
            print(f"✅ Model loaded successfully on device: {DEVICE}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    return _wav2vec_model, _wav2vec_processor


def extract_wav2vec_features(file_path, target_sr=16000, max_duration=2):
    """
    Extract wav2vec 2.0 embeddings from an audio file.
    
    Args:
        file_path: Path to WAV file
        target_sr: Target sample rate for wav2vec (typically 16000 Hz)
        max_duration: Maximum duration in seconds
        
    Returns:
        numpy array of shape (sequence_length, hidden_dim)
    """
    model, processor = load_wav2vec_model()
    
    # Load audio at the target sample rate
    waveform, sr = librosa.load(file_path, sr=target_sr)
    
    # Trim to max duration
    max_samples = target_sr * max_duration
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    
    # Pad if shorter
    if len(waveform) < max_samples:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode='constant')
    
    # Process with wav2vec
    inputs = processor(waveform, sampling_rate=target_sr, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract last hidden state (sequence of embeddings)
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # Shape: (sequence_length, 768)
    
    return embeddings


def get_pooled_wav2vec_features(file_path, pooling="mean"):
    """
    Extract wav2vec features and pool them into a single vector.
    Useful for fixed-size input to a classifier.
    
    Args:
        file_path: Path to WAV file
        pooling: Pooling method - "mean", "max", or "concat"
        
    Returns:
        numpy array of shape (embedding_dim,) or (embedding_dim * num_layers,)
    """
    embeddings = extract_wav2vec_features(file_path)  # Shape: (seq_len, 768)
    
    if pooling == "mean":
        pooled = np.mean(embeddings, axis=0)
    elif pooling == "max":
        pooled = np.max(embeddings, axis=0)
    elif pooling == "concat":
        # Mean and max concatenation
        pooled = np.concatenate([np.mean(embeddings, axis=0), np.max(embeddings, axis=0)])
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
    
    return pooled


def extract_batch_wav2vec_features(file_paths, pooling="mean"):
    """
    Extract wav2vec features for multiple files.
    
    Args:
        file_paths: List of file paths
        pooling: Pooling method
        
    Returns:
        numpy array of shape (num_files, embedding_dim)
    """
    features = []
    for i, path in enumerate(file_paths):
        try:
            feat = get_pooled_wav2vec_features(path, pooling=pooling)
            features.append(feat)
            if (i + 1) % 10 == 0:
                print(f"✅ Processed {i + 1}/{len(file_paths)} files")
        except Exception as e:
            print(f"⚠️  Error processing {path}: {e}")
            # Return a zero vector of the right size
            dummy_feat = get_pooled_wav2vec_features(file_paths[0], pooling=pooling)
            features.append(np.zeros_like(dummy_feat))
    
    return np.array(features)


def get_wav2vec_embeddings_for_gradcam(file_path, reshape_to_spectrogram=False):
    """
    Extract wav2vec embeddings and optionally reshape to match spectrogram shape.
    This allows wav2vec to work with Grad-CAM visualizations.
    
    Args:
        file_path: Path to WAV file
        reshape_to_spectrogram: If True, reshape to (1, 128, 130, 1) to match Mel-spectrogram
        
    Returns:
        numpy array, optionally reshaped
    """
    embeddings = extract_wav2vec_features(file_path)  # (seq_len, 768)
    
    if reshape_to_spectrogram:
        # Reshape to match Mel-spectrogram shape (1, 128, 130, 1)
        # This is a lossy conversion but allows compatibility with existing Grad-CAM code
        seq_len = embeddings.shape[0]
        
        # Resize to (128, 130) and add batch and channel dims
        reshaped = embeddings[:128*130].reshape(128, 130, 1)
        reshaped = np.pad(reshaped, ((0, 0), (0, 0), (0, 0)), mode='constant')
        reshaped = np.expand_dims(reshaped, 0)  # Add batch dimension
        
        return reshaped
    
    return embeddings


if __name__ == "__main__":
    # Example usage
    print("🎵 wav2vec 2.0 Feature Extractor")
    print("=" * 50)
    
    # Example: Extract features from a sample file
    sample_file = "temp.wav"  # Replace with actual file
    
    try:
        features = get_pooled_wav2vec_features(sample_file, pooling="mean")
        print(f"✅ Extracted features shape: {features.shape}")
        print(f"   Feature vector dim: {features.shape[0]}")
    except Exception as e:
        print(f"⚠️  Could not extract features: {e}")
