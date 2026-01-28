import gradio as gr
import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
import keras

# --- Config ---
# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
IMAGE_FOLDER = os.path.join(BASE_DIR, "test_images")
IMG_SIZE = (128, 128)

# --- Global State Helpers ---
loading_error = None

# MONKEY PATCH: Handle 'quantization_config' in Dense layer
# This patch ensures that if 'quantization_config' is passed to Dense layer (from saved model),
# it is safely ignored instead of raising an error.
try:
    _original_dense_init = keras.layers.Dense.__init__
    def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
        _original_dense_init(self, *args, **kwargs)
    keras.layers.Dense.__init__ = _patched_dense_init
    print("✅ Applied monkey patch to keras.layers.Dense for quantization_config compatibility")
except Exception as e:
    print(f"⚠️ Failed to patch Dense layer: {e}")

def load_game_resources():
    global loading_error
    """Loads model and image list."""
    
    print(f"DEBUG: Keras Version: {keras.__version__}")
    print(f"DEBUG: TensorFlow Version: {tf.__version__}")

    # Load Model
    try:
        # compile=False avoids errors related to optimizers/losses that aren't needed for inference
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        loading_error = str(e)
        model = None

    # Load Images
    images = []
    if os.path.exists(IMAGE_FOLDER):
        for fname in os.listdir(IMAGE_FOLDER):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(IMAGE_FOLDER, fname)
                # Determine label from filename prefix created in prepare_assets.py
                label = "Real" if "real_" in fname.lower() else "AI"
                images.append((path, label))
    
    random.shuffle(images)
    print(f"✅ Loaded {len(images)} images.")
    return model, images

# Load resources once on startup
global_model, global_images = load_game_resources()

# --- Game Logic ---

def start_new_game():
    """Reset scores and reshuffle images."""
    random.shuffle(global_images)
    # State: [human_score, model_score, round_count, current_image_index, image_list]
    return 0, 0, 0, 0, global_images, gr.update(visible=True), gr.update(visible=False)

def get_current_image(index, image_list):
    if not image_list or index >= len(image_list):
        return None, "Game Over! Restart to play again.", None
    
    path, label = image_list[index]
    return path, f"Round {index + 1}", label

def predict_and_score(user_guess, index, image_list, human_score, model_score, rounds):
    if not image_list or index >= len(image_list):
        return human_score, model_score, rounds, index, "Game Over", None, gr.update(visible=False), gr.update(visible=True)

    path, true_label = image_list[index]
    
    # 1. Model Prediction
    model_guess_label = "Unsure"
    confidence = 0.0
    
    if global_model:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) # Model has internal Rescaling layer, so keep [0, 255]
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction (Sigmoid: <0.5 = Class 0, >0.5 = Class 1)
            # We need to know which class is which. 
            # Usually: 0=AI, 1=Real OR 0=Real, 1=AI.
            # In the notebook: train_generator.class_indices would tell us.
            # Assumption based on directory names: sorting usually puts AI first.
            # So 0=AI, 1=Real. Let's verify this output if possible, but assume 0=AI for now.
            prediction = global_model.predict(img_array, verbose=0)[0][0]
            
            # Notebook typically uses 'training_AI' and 'training_real'. 
            # Alphabetical: AI=0, real=1.
            is_real = prediction > 0.5
            model_guess_label = "Real" if is_real else "AI"
            confidence = prediction if is_real else 1 - prediction
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            model_guess_label = "Error"
    else:
        model_guess_label = f"Load Error: {loading_error}"
    
    # 2. Update Scores
    human_correct = (user_guess == true_label)
    model_correct = (model_guess_label == true_label)
    
    if human_correct:
        human_score += 1
    if model_correct:
        model_score += 1
        
    rounds += 1
    next_idx = index + 1
    
    # ... (Previous code)
    
    # 3. Message & Visuals
    if human_correct:
        # Green success text
        result_text = f"## <span style='color:green'>Correct! It was {true_label}. 🎉</span>\n\n"
        # Trigger confetti JS
        js_cmd = """
            <script>
            confetti({
                particleCount: 100,
                spread: 70,
                origin: { y: 0.6 }
            });
            </script>
        """
    else:
        # Red failure text
        result_text = f"## <span style='color:red'>Wrong! It was {true_label}. 😢</span>\n\n"
        js_cmd = ""

    result_text += f"**You guessed:** {user_guess}\n"
    
    model_result_color = "green" if model_correct else "red"
    result_text += f"**Model guessed:** <span style='color:{model_result_color}'>{model_guess_label}</span> ({confidence:.2f})"

    # Prepare next round view
    # Hide guess buttons, Show Next button
    return human_score, model_score, rounds, next_idx, result_text, gr.update(visible=False), gr.update(visible=True), js_cmd

# Helper for Game Over HTML
def get_game_over_html(human_score, model_score):
    winner = "It's a Tie!"
    color = "#6b7280" # Gray
    if human_score > model_score:
        winner = "🎉 YOU WON! 🎉"
        color = "#10b981" # Green
    elif model_score > human_score:
        winner = "🤖 AI WON! 🤖"
        color = "#ef4444" # Red
        
    return f"""
    <div style="text-align: center; animation: fadeIn 2s; padding: 20px;">
        <style>
            @keyframes fadeIn {{
                0% {{ opacity: 0; transform: translateY(-20px); }}
                100% {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
        <h1 style="font-size: 4em; color: {color}; margin-bottom: 10px;">GAME OVER</h1>
        <h2 style="font-size: 2.5em; margin-top: 0;">{winner}</h2>
        <div style="font-size: 2em; margin: 20px 0; border: 2px solid {color}; display: inline-block; padding: 10px 30px; border-radius: 10px;">
            Human: {human_score} &nbsp;|&nbsp; AI: {model_score}
        </div>
        <p style="font-size: 1.2em; opacity: 0.8;">Click 'Restart Game' to play again</p>
    </div>
    """

def next_round_view(index, image_list, human_score, model_score):
    path, round_text, _ = get_current_image(index, image_list)
    if path is None:
         # End game - Show big visual summary
         final_html = get_game_over_html(human_score, model_score)
         return None, final_html, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), "" 
    
    return path, round_text, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), ""

# --- UI Construction ---
with gr.Blocks() as demo:
    
    # Load Confetti Library
    gr.HTML("""
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
    """)

    # Game State
    # Stores: [human_score, model_score, rounds, index, image_list]
    state_human = gr.State(0)
    state_model = gr.State(0)
    state_rounds = gr.State(0)
    state_index = gr.State(0)
    state_images = gr.State(global_images)

    gr.Markdown("# 🤖 Human vs. AI: Face Detection Showdown")
    gr.Markdown("Can you beat the deep learning model at spotting AI-generated faces?")
    
    with gr.Row():
        # Left Column: Stats
        with gr.Column(scale=1):
            human_score_disp = gr.Number(label="Your Score", value=0)
            model_score_disp = gr.Number(label="Model Score", value=0)
            round_disp = gr.Number(label="Rounds Played", value=0)
            restart_btn = gr.Button("🔄 Restart Game", variant="secondary")

        # Right Column: Game Area
        with gr.Column(scale=2):
            round_label = gr.Markdown("### Round 1")
            image_display = gr.Image(label="Is this face Real or AI?", type="filepath", height=300)
            
            # Result Display
            result_box = gr.Markdown("")
            
            # Visual Effects Trigger (Hidden HTML component to run JS)
            js_trigger = gr.HTML(visible=False)
            
            with gr.Row() as guess_row:
                btn_real = gr.Button("Real 👤", variant="primary", size="lg")
                btn_ai = gr.Button("AI 🤖", variant="primary", size="lg")
            
            next_btn = gr.Button("Next Image ➡️", variant="primary", visible=False)

    # Instructions & Details
    with gr.Accordion("ℹ️ How to Play & Model Details", open=False):
        gr.Markdown("""
        **How to Play:**
        1. Look at the image and decide if it's a **Real** person or **AI-Generated**.
        2. Click the corresponding button.
        3. See if you can beat the AI!

        **Understanding the Score:**
        - The number in brackets, e.g., `(0.95)`, represents the **Model's Confidence Percentage** in its guess (95% confident).
        
        **Model Details:**
        - This game uses our **Best Fine-Tuned MobileNetV2 Model**, which achieved the highest accuracy during training.
        - It has been trained to detect subtle artifacts in AI-generated faces.
        """)

    # --- Event Handlers ---
    
    # Restart
    restart_btn.click(
        start_new_game,
        outputs=[state_human, state_model, state_rounds, state_index, state_images, guess_row, next_btn]
    ).then(
        next_round_view,
        inputs=[state_index, state_images, state_human, state_model],
        outputs=[image_display, round_label, guess_row, next_btn, restart_btn, js_trigger]
    ).then(
        lambda: (0,0,0, ""), outputs=[human_score_disp, model_score_disp, round_disp, result_box]
    )

    # Guess Real
    btn_real.click(
        fn=predict_and_score,
        inputs=[gr.State("Real"), state_index, state_images, state_human, state_model, state_rounds],
        outputs=[state_human, state_model, state_rounds, state_index, result_box, guess_row, next_btn, js_trigger]
    ).then(
        lambda h, m, r: (h, m, r),
        inputs=[state_human, state_model, state_rounds],
        outputs=[human_score_disp, model_score_disp, round_disp]
    )
    
    # Guess AI
    btn_ai.click(
        fn=predict_and_score,
        inputs=[gr.State("AI"), state_index, state_images, state_human, state_model, state_rounds],
        outputs=[state_human, state_model, state_rounds, state_index, result_box, guess_row, next_btn, js_trigger]
    ).then(
        lambda h, m, r: (h, m, r),
        inputs=[state_human, state_model, state_rounds],
        outputs=[human_score_disp, model_score_disp, round_disp]
    )

    # Next Image
    next_btn.click(
        next_round_view,
        inputs=[state_index, state_images, state_human, state_model],
        outputs=[image_display, round_label, guess_row, next_btn, restart_btn, js_trigger]
    ).then(
        lambda: "", outputs=[result_box]
    )

    # Initial Load
    demo.load(
        start_new_game,
        outputs=[state_human, state_model, state_rounds, state_index, state_images, guess_row, next_btn]
    ).then(
        next_round_view,
        inputs=[state_index, state_images, state_human, state_model],
        outputs=[image_display, round_label, guess_row, next_btn, restart_btn, js_trigger]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)
