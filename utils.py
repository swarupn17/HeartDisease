import librosa
import numpy as np

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # 🔹 FIXED duration (IMPORTANT)
    target_length = 22050 * 2  # 2 seconds (matches most PCG training)

    if len(y) < target_length:
        # 🔥 FIX: Instead of zero-padding (silence), loop the audio signal
        # This eliminates the blank right-side of the spectrogram by replacing
        # silence with real heartbeat data
        num_full_loops = target_length // len(y)
        remainder = target_length % len(y)
        y = np.tile(y, num_full_loops + 1)[:target_length]
    else:
        y = y[:target_length]

    # 🔹 NO aggressive normalization (important!)
    # keep signal closer to original training scale

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 🔥 CRITICAL: DO NOT STANDARDIZE
    # remove this if you added earlier:
    # mel_db = (mel_db - mean)/std ❌

    # 🔹 Fix width
    target_width = 130

    if mel_db.shape[1] < target_width:
        # 🔥 FIX: Loop the spectrogram time steps instead of zero-padding
        # This ensures the right side contains real data, not dead silence
        num_full_loops = target_width // mel_db.shape[1]
        remainder = target_width % mel_db.shape[1]
        mel_db = np.tile(mel_db, (1, num_full_loops + 1))[:, :target_width]
    else:
        mel_db = mel_db[:, :target_width]

    # 🔹 Final shape
    mel_db = np.expand_dims(mel_db, -1)
    mel_db = np.expand_dims(mel_db, 0)

    return mel_db