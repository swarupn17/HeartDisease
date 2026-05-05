import os
from dotenv import load_dotenv

# 🔹 Load environment variables from .env file
load_dotenv()

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

# 🔥 Fix Mac matplotlib crash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI not installed. LLM explanations will be skipped.")

from utils import preprocess_audio
from gradcam import get_gradcam
from gradcam import get_last_conv_layer_name, save_superimposed_gradcam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
REPORT_IMAGES_DIR = os.path.join(STATIC_DIR, "report_images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
FRONTEND_DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
FRONTEND_INDEX_FILE = os.path.join(FRONTEND_DIST_DIR, "index.html")

os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# 🔹 Load models
binary_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "final_cnn_transformer_binary.keras"),
    compile=False
)

sub_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "final_cnn_transformer_subclass.keras"),
    compile=False
)

# 🔹 IMPORTANT: Verify this order matches training labels
LABELS = ["Normal", "Murmur", "Extrastole", "Artifact"]

# 🔹 Initialize Gemini API if available
GEMINI_MODELS = []  # List of available generative models
if GEMINI_AVAILABLE:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        print("✅ Gemini API initialized")
        # Discover available models
        try:
            models = genai.list_models()
            # Filter for generative models (exclude embedding, image generation, etc.)
            generative_models = [m.name.split('/')[-1] for m in models if 'generateContent' in m.supported_generation_methods]
            if generative_models:
                GEMINI_MODELS = generative_models
                print(f"📋 Available generative models: {generative_models}")
                print(f"🤖 Will use model: {generative_models[0]}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}. Will attempt fallback models.")
    else:
        GEMINI_AVAILABLE = False
        print("⚠️  GEMINI_API_KEY environment variable not set")


def generate_clinical_explanation(predicted_class, confidence, time_seconds=2):
    """
    Use Gemini to generate a clinical cardiologist-style explanation.
    Falls back to template if API unavailable.
    """
    if not GEMINI_AVAILABLE or not GEMINI_MODELS:
        # Fallback explanation
        return f"The model detected {predicted_class} with {confidence:.1%} confidence across the {time_seconds}-second cardiac cycle."
    
    try:
        prompt = f"""You are an expert cardiologist reviewing a cardiac sound analysis with visual Grad-CAM explainability. 

Predicted condition: {predicted_class}
Confidence: {confidence:.1%}
Recording duration: {time_seconds} seconds

Provide a 2-sentence clinical interpretation suitable for a cardiology report. You MUST explicitly reference the Grad-CAM heatmap visualization and explain what acoustic anomaly (visual evidence) the model is highlighting that led to this diagnosis.

Guidelines:
- Do NOT mention spectrogram, bins, frames, CNN, or machine learning details
- MUST connect the visual heatmap evidence to the clinical diagnosis
- Describe the acoustic anomaly visible in the heatmap in clinical terms (e.g., "anomalous high-energy pulse", "prolonged murmur signature", "abnormal frequency concentration")
- Use medical terminology:
  * Refer to frequency regions as acoustic signatures (e.g., "low-frequency acoustic signature")
  * Refer to time intervals as cardiac phases (e.g., "systolic phase", "diastolic phase")
  * Reference cardiac structures when relevant (e.g., "S1/S2 cardiac cycle", "ventricular phase")
- Act as a "tour guide" explaining why the heatmap visualization supports this specific diagnosis
- Focus on clinical significance and actionable next steps

Example: "The model detected a high-confidence acoustic signature consistent with an extrastolic beat. As highlighted in the Grad-CAM visualization, there is an anomalous, high-energy acoustic pulse occurring prematurely before the anticipated S1/S2 cardiac cycle. Though noted in a brief 2-second window, this premature contraction necessitates correlation with a full ECG assessment to determine its morphology and clinical significance."

Keep it concise, clinically rigorous, and directly connected to the visual evidence."""
        
        # Try primary model first, then fallback models
        models_to_try = GEMINI_MODELS + ["gemini-pro", "models/gemini-pro"]
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                result = response.text.strip()
                print(f"✅ Gemini generated explanation using {model_name}")
                return result
            except Exception as e:
                continue
        
        # If all models failed, return fallback
        print(f"⚠️  All Gemini models failed")
        return f"The model detected {predicted_class} with {confidence:.1%} confidence."
        
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        return f"The model detected {predicted_class} with {confidence:.1%} confidence."


def analyze_audio_file(path):
    """Run inference, build Grad-CAM, and return a structured response."""
    # 🔹 Preprocess
    x = preprocess_audio(path)

    print("\n--- DEBUG INFO ---")
    print("Input shape:", x.shape)

    # 🔹 Predictions
    binary_pred = float(binary_model.predict(x)[0][0])
    sub_probs = sub_model.predict(x)[0]
    sub_class = int(np.argmax(sub_probs))
    sub_conf = float(np.max(sub_probs))
    sub_label = LABELS[sub_class]

    result = "Normal" if sub_label == "Normal" else "Abnormal"

    print("Binary:", binary_pred)
    print("Subclass probs:", sub_probs)
    print("Subclass index:", sub_class)
    print("Subclass confidence:", sub_conf)

    if binary_pred > 0.5:
        result = "Abnormal"
    elif sub_conf > 0.75 and sub_class != 0:
        result = "Abnormal"
    else:
        result = "Normal"

    print("Final Result:", result)
    print("------------------\n")

    # 🔹 Extract original spectrogram dimensions for Grad-CAM alignment
    spec_height = x.shape[1]
    spec_width = x.shape[2]

    print(f"\n📊 SPECTROGRAM INFO:")
    print(f"   Full input shape: {x.shape}")
    print(f"   Spectrogram height (freq bins): {spec_height}")
    print(f"   Spectrogram width (time steps): {spec_width}")
    print(f"   Squeezed spectrogram shape for plotting: {x[0].squeeze().shape}\n")

    # 🔹 Save raw spectrogram for debugging upstream padding/clipping
    raw_output_name = "report_images/raw_input.png"
    raw_output_path = os.path.join(STATIC_DIR, raw_output_name)
    plt.figure(figsize=(8, 4))
    plt.imshow(x[0].squeeze(), cmap='gray', aspect='auto', origin='lower')
    plt.axis("off")
    plt.savefig(raw_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"📊 RAW SPECTROGRAM SAVED: {raw_output_path}\n")

    # 🔹 Grad-CAM with the last convolutional layer
    last_conv_layer_name = get_last_conv_layer_name(binary_model)
    print(f"📊 Using Grad-CAM layer: {last_conv_layer_name}")
    heatmap = get_gradcam(binary_model, x, last_conv_layer_name)

    # 🔹 Save visualization
    print(f"📊 HEATMAP BEFORE SAVE: shape={heatmap.shape}\n")

    output_name = "report_images/output.png"
    output_path = os.path.join(STATIC_DIR, output_name)
    save_superimposed_gradcam(x, heatmap, output_path=output_path)

    confidence_breakdown = [
        {"label": label, "value": float(prob * 100)}
        for label, prob in zip(LABELS, sub_probs)
    ]

    # 🔹 Generate clinical explanation using LLM
    explanation = generate_clinical_explanation(sub_label, sub_conf, time_seconds=2)

    return {
        "result": result,
        "subclass": sub_label,
        "confidence": sub_conf,
        "binary_confidence": binary_pred,
        "confidence_breakdown": confidence_breakdown,
        "raw_image_url": url_for('static', filename=raw_output_name),
        "image_url": url_for('static', filename=output_name),
        "explanation": explanation,
    }


def frontend_is_built():
    return os.path.exists(FRONTEND_INDEX_FILE)


@app.route("/assets/<path:filename>")
def frontend_assets(filename):
    if frontend_is_built():
        return send_from_directory(os.path.join(FRONTEND_DIST_DIR, "assets"), filename)

    return jsonify({"error": "Frontend build not found"}), 404


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if not file:
            return render_template("index.html")

        path = os.path.join(BASE_DIR, "temp.wav")
        file.save(path)
        analysis = analyze_audio_file(path)

        return render_template(
            "index.html",
            result=analysis["result"],
            subclass=analysis["subclass"],
            image="output.png"
        )

    if frontend_is_built():
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")

    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    path = os.path.join(BASE_DIR, "temp.wav")
    file.save(path)

    analysis = analyze_audio_file(path)
    return jsonify(analysis)


if __name__ == "__main__":
    app.run(debug=True, port=5001)