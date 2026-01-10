
import base64
import streamlit as st
from PIL import Image
import os
import json
import numpy as np
from datetime import datetime
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input



# === Setup ===
st.set_page_config("Monkeypox Detection App")
UPLOAD_FOLDER = "uploads"
USER_FILE = "users.json"
HISTORY_FILE = "history.json"

# === Create folders ===
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("utils", exist_ok=True)

# === Load users ===
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# === Load history ===
def load_history():
    history_path = "history.json"  # update with your path if needed

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            try:
                data = f.read().strip()
                return json.loads(data) if data else {}  # return empty dict if file is empty
            except json.JSONDecodeError:
                print("⚠️ JSON file is corrupted. Starting with empty history.")
                return {}
    else:
        return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

users = load_users()
history = load_history()
import tensorflow as tf

# Load the saved model directory (not .keras file!)
model = tf.saved_model.load('ensemble_model_tf')

# === Session State ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# === Auth Section ===
st.sidebar.title("🔐 Login / Signup")

mode = st.sidebar.radio("Choose:", ["Login", "Signup"])
if mode == "Signup":
    full_name = st.sidebar.text_input("Full Name")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    confirm = st.sidebar.text_input("Confirm Password", type="password")
    if st.sidebar.button("Create Account"):
        if username in users:
            st.sidebar.warning("Username already exists.")
        elif password != confirm:
            st.sidebar.warning("Passwords do not match.")
        else:
            users[username] = {"full_name": full_name, "password": password}
            save_users(users)
            st.sidebar.success("Account created! You can login now.")

if mode == "Login":
    username = st.sidebar.text_input("Username", key="login_user")
    password = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.sidebar.success(f"Welcome {users[username]['full_name']}!")
        else:
            st.sidebar.error("Invalid credentials.")

if st.session_state.logged_in:
    st.sidebar.markdown(f"✅ Logged in as: {st.session_state.current_user}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.experimental_rerun()

# === Main App ===
st.title("🧪 Monkeypox Detection using Ensemble Models")
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_monkeypox_folder(folder_path):
    """
    Predicts monkeypox presence for all images in a folder OR a single image.

    Parameters:
    - folder_path (str): Path to a folder containing images OR a single image path

    Returns:
    - List of prediction results
    """
    img_size = 224
    results = []

    # Check if it's a single image file
    if os.path.isfile(folder_path):
        image_paths = [folder_path]  # Treat single image as a list
    else:
        # Get all files in the folder, filtering for valid images
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for img_path in image_paths:
        img_name = os.path.basename(img_path) # Get the image file name
        try:
            # Load image and convert to NumPy array
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  
            with open('ensemble.pkl','rb') as f:
                mp = pickle.load(f)
                prediction = mp.predict(img_array)[0]  # Get raw prediction
            label = "Monkeypox" if prediction.any() > 0.5 else "No Monkeypox"
            confidence = np.max(prediction) if prediction.any() > 0.5 else 1 - np.max(prediction) 

            results.append({
                "Image": img_name,
                "Prediction": label,
                "Confidence": round(confidence, 2),
                "prediction": prediction.tolist(),
            })

            print(f"{img_name}: {label} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

    return results

def predict_single_image(img_path, model, img_size=128):
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Call the default serving function
        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_array)
        raw_pred = infer(input_tensor)

        # Extract prediction from output dict
        key = list(raw_pred.keys())[0]
        prediction = raw_pred[key].numpy()[0][0]

        label = "No Monkeypox" if prediction > 0.5 else "Monkeypox"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        print(f"{os.path.basename(img_path)} => {label} (Confidence: {confidence:.2f})")
        return label

    except Exception as e:
        print(f"Error: {e}")
        return None, None


if st.session_state.logged_in:
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)


        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        image.save(save_path)
        

        if st.button("Predict"):
            label = predict_single_image(save_path,model)
            print(label)  # Call ensemble prediction function
            st.success(f"Prediction: **{label}**")

            # Save to history
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user = st.session_state.current_user
            if user not in history:
                history[user] = []
            history[user].append({"image": save_path, "prediction": label, "time": now})
            save_history(history)

    # Show Prediction History
    st.subheader("📜 Your Prediction History")
    user_history = history.get(st.session_state.current_user, [])
    for entry in reversed(user_history):
        st.image(entry["image"], width=100)
        st.write(f"🕒 {entry['time']} | 🧾 {entry['prediction']}")
else:
    st.warning("Please log in to access the app.")

