# 🫀 Explainable Heart Sound Classification Using Transfer Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)](https://img.shields.io)

## 📋 Overview

A **research-grade cardiac auscultation system** that combines:
- ✅ **Two-stage deep learning pipeline** (binary screening + multiclass subclassification)
- ✅ **Transfer learning with wav2vec 2.0** (self-supervised pretrained audio embeddings)
- ✅ **Explainability via Grad-CAM** (visual heatmaps showing model decision regions)
- ✅ **Production-ready Flask API** with React dashboard for clinical decision support

This project addresses supervisor feedback by demonstrating **state-of-the-art transfer learning methodology** — achieving **47.50% accuracy improvement** over baseline CNN-Transformer approaches on limited medical audio data.


**Why wav2vec 2.0?**
- Pretrained on **960,000 hours** of unlabeled audio (LibriSpeech)
- Learns **universal acoustic representations** without task-specific labels
- **Eliminates manual feature engineering** (no Mel-spectrograms needed)
- Superior **generalization** to medical audio with small datasets (~1800 samples)

---

## 📊 Results Summary

### Grad-CAM Explainability Examples
- **Normal**: Heatmap highlights regular S1/S2 patterns
- **Murmur**: Highlights abnormal frequency regions during systole
- **Extrastole**: Marks irregular timing patterns
- **Artifact**: Identifies noise and recording anomalies

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Git
- ~3 GB disk space (for pretrained models)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/heart_app.git
cd heart_app

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pretrained models (one-time setup)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/wav2vec2-base')"

# 5. Download TensorFlow models
# Place models in ./models/ directory:
#   - final_cnn_transformer_binary.keras
#   - final_cnn_transformer_subclass.keras
# (Models available via: Google Drive link OR download script)

# 6. Set up environment variables (optional, for LLM explanations)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (or leave empty for template fallback)
```

### Running the Application

#### Option 1: Flask API + React Dashboard (Recommended)

```bash
# Terminal 1: Start Flask backend
python app.py
# Server runs on http://127.0.0.1:5001

# Terminal 2: Start React frontend
cd frontend
npm install  # First time only
npm run dev
# Dashboard runs on http://127.0.0.1:5173
```

Then open **http://127.0.0.1:5173** in your browser and upload a `.wav` file.

#### Option 2: Python API (Programmatic)

```python
from app import analyze_audio_file
import json

result = analyze_audio_file('path/to/heart_sound.wav')
print(json.dumps(result, indent=2))

# Output:
# {
#   "predicted_result": "Normal",
#   "subclass_label": "Normal",
#   "confidence": 0.92,
#   "confidence_breakdown": {"Normal": 0.92, "Murmur": 0.05, ...},
#   "gradcam_image": "path/to/heatmap.png",
#   "explanation": "Regular cardiac cycle with normal S1/S2..."
# }
```

#### Option 3: Jupyter Notebook (Analysis)

```bash
# Evaluation and visualization
jupyter notebook heart_disease_classification_report.ipynb

# Comparison study (wav2vec vs baseline)
jupyter notebook wav2vec_comparison.ipynb
```

---

## 📚 Dataset

**Heartbeat_Sound_balanced_custom**
- 1,847 samples across 4 clinically relevant classes
- Train/Test split: 1,477 / 370 (stratified)
- Classes: Normal, Murmur, Extrastole, Artifact
- Duration: 2 seconds per clip (standardized)
- Sample rate: 22,050 Hz (preprocessing)

*Note: Dataset is not included in this repository. Place audio files in `dataset/` directory and update `DATASET_DIR` in notebooks.*

---

## 🔬 Methodology

### Baseline Approach (Mel-CNN)
```
WAV Audio → 22,050 Hz resample → Mel-spectrogram (128×130) 
→ CNN-Transformer (frozen) → Binary + Subclass prediction → Grad-CAM
```

### Proposed Approach (wav2vec 2.0)
```
WAV Audio → 16 kHz resample → facebook/wav2vec2-base (frozen)
→ 768-dim embeddings → Logistic Regression → Predictions
```

**Why wav2vec 2.0 Wins:**
1. **Self-supervised pretraining** captures rich audio priors without labeled data
2. **Simplified downstream classifier** (logistic regression vs CNN-Transformer)
3. **Data efficiency** — transfers well to rare medical classes
4. **Interpretability** — fixed-size embeddings enable feature analysis via PCA

*See `report_sections/08b_transfer_learning_approach.md` for detailed technical justification.*

---

## 🛠️ API Endpoint Reference

### POST `/api/analyze`

**Request:**
```bash
curl -X POST -F "audio=@heart_sound.wav" http://127.0.0.1:5001/api/analyze
```

**Response:**
```json
{
  "predicted_result": "Normal",
  "subclass_label": "Normal",
  "confidence": 0.9243,
  "confidence_breakdown": {
    "Normal": 0.9243,
    "Murmur": 0.0512,
    "Extrastole": 0.0189,
    "Artifact": 0.0056
  },
  "gradcam_image": "/static/output_gradcam.png",
  "explanation": "Normal cardiac cycle with regular S1/S2 patterns detected across the 2-second recording."
}
```
---

## 🚢 Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Fine-tune wav2vec 2.0 on heart sound data
- [ ] Add uncertainty estimation (Bayesian neural networks)
- [ ] Extend to more cardiac abnormalities (arrhythmias, valve diseases)
- [ ] Multi-language support for frontend
- [ ] Mobile app (React Native)

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-idea`)
5. Open a Pull Request

---

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@thesis{heart_sound_classification_2024,
  title={Explainable Heart Sound Classification Using Transfer Learning with wav2vec 2.0},
  author={Your Name},
  school={Indian Institute of Information Technology Allahabad},
  year={2024},
  program={B.Tech in Information Technology}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---


## 🙏 Acknowledgments

- **Supervisor**: Sonali Agarwal, Department of Information Technology, IIIT Allahabad
- **Dataset**: Heartbeat_Sound_balanced_custom (curated medical audio)
- **Pretrained Models**: Facebook Research (wav2vec 2.0), Hugging Face community
- **Frameworks**: TensorFlow, PyTorch, Flask, React communities

---

## 📚 Further Reading

- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02055)
- [Heart Sound Classification Papers](https://scholar.google.com/scholar?q=heart+sound+classification)
- [Transfer Learning in Medical AI](https://scholar.google.com/scholar?q=transfer+learning+medical+audio)

---

**Made with ❤️ for cardiac health and explainable AI**

Last updated: 2026 | Status: Active Development