import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessor
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("kmeans_model.pkl")

# Page setup
st.set_page_config(page_title="🎓 SNU Study Buddy Finder", page_icon="🎯", layout="centered")

# --- BEAUTIFUL CUSTOM CSS ---
st.markdown("""
    <style>
        /* Background image */
        [data-testid="stAppViewContainer"] {
            background-image: url('https://images.unsplash.com/photo-1503676260728-1c00da094a0b');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }

        /* Glassmorphism card */
        .stApp {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.2);
        }

        /* Title */
        h1 {
            text-align: center;
            background: linear-gradient(to right, #0066ff, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: 800;
        }

        /* Buttons */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
            border: none;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #0052cc 0%, #0099cc 100%);
        }

        /* Form labels */
        label {
            font-weight: bold !important;
            color: #003366 !important;
        }

        /* Result card */
        .success {
            background: rgba(0, 204, 153, 0.1);
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #00cc99;
            color: #004d40;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 SNU Study Buddy Finder")
st.markdown("### Find your ideal study partner based on personality and interests!")

# --- INPUT FORM ---
with st.form("study_buddy_form"):
    age = st.selectbox("Age Group", ["18", "19", "20", "21", "22", "23"])
    books = st.selectbox("Books read past year", ["0-5", "6-10", "11-20", "20+"])
    intro_extro = st.slider("Introversion ↔ Extraversion", 1, 10, 5)
    club = st.selectbox("Club/Organization", ["Coding", "Music", "Art", "Sports", "Drama", "None"])
    submit = st.form_submit_button("🔍 Find My Study Buddy Group")

# --- PREDICTION ---
if submit:
    user_df = pd.DataFrame({
        'Age': [age],
        'Books read past year Provide in integer value between (0-50)': [books],
        'Introversion extraversion': [intro_extro],
        'Club top1': [club]
    })

    user_X = preprocessor.transform(user_df)
    cluster = model.predict(user_X)[0]

    st.markdown(
        f"<div class='success'><h3>🎯 You belong to Study Buddy Group #{cluster}</h3>"
        f"<p>Students in this group share your teamwork style and study habits. "
        f"Collaborate to make learning more productive and fun! ✨</p></div>",
        unsafe_allow_html=True
    )
