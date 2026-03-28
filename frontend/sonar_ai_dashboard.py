# sonar_ai_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ------------------------------
# Load Model / Assets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")
model = joblib.load(MODEL_PATH)

# ------------------------------
# Assets
ROCK_IMG = os.path.join(BASE_DIR, "assets", "rock.jpg")
MINE_IMG = os.path.join(BASE_DIR, "assets", "mine.jpg")
LOGO_IMG = os.path.join(BASE_DIR, "assets", "logo.png")

# ------------------------------
# Session State
if "user_logged_in" not in st.session_state:
    st.session_state["user_logged_in"] = True
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# ------------------------------
# CSS Styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #0B3D91;
        color: white;
    }
    .stButton>button {
        background-color: #0B3D91;
        color: white;
        border-radius: 8px;
    }
    div[data-testid="stImage"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .center-text {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# ------------------------------
# Helper Functions
def predict_sonar(input_dict):
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    conf = model.predict_proba(df).max()
    return pred, conf

# Sidebar metrics
def sidebar_metrics():
    total = len(st.session_state["predictions"])
    rocks = sum(1 for p in st.session_state["predictions"] if p["prediction"]=="R")
    mines = sum(1 for p in st.session_state["predictions"] if p["prediction"]=="M")
    avg_conf = 0
    if total>0:
        avg_conf = sum(p["confidence"] for p in st.session_state["predictions"])/total

    st.sidebar.markdown("## 📊 Dashboard Metrics")
    st.sidebar.markdown(f"**Total Predictions:** {total}")
    st.sidebar.markdown(f"**Rocks:** {rocks}")
    st.sidebar.markdown(f"**Mines:** {mines}")
    st.sidebar.markdown(f"**Avg Confidence:** {avg_conf*100:.2f}%")

# ------------------------------
# Pages
def login_page():
    st.image(LOGO_IMG, width=250)
    st.markdown('<h1 class="center-text">SONAR AI Dashboard Login</h1>', unsafe_allow_html=True)
    password = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        if password == "sonar123":  # simple demo password
            st.session_state["user_logged_in"] = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Incorrect password!")

def home_page():
    st.image(LOGO_IMG, width=300)
    st.markdown(
        """
        <div class="center-text">
            <h2>Welcome to SONAR AI Dashboard</h2>
            <p>This dashboard predicts <strong>Rock vs Mine</strong> using a trained Logistic Regression model.</p>
            <p>Navigate using the sidebar to make predictions, view analytics, and see your prediction history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def predict_page():
    st.header("Predict Rock vs Mine")
    st.write("Enter 60 sonar feature values (0–1 range):")
    input_data = {}
    for i in range(1, 61):
        input_data[f"V{i}"] = st.number_input(f"V{i}", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict"):
        pred, conf = predict_sonar(input_data)
        st.session_state["predictions"].append({
            "input": input_data,
            "prediction": pred,
            "confidence": conf
        })

        # Display prediction as colored card with image
        if pred == "R":
            st.markdown(f"""
            <div style="padding:20px; background-color:#4CAF50; color:white; border-radius:10px; text-align:center;">
                <h2>Prediction: ROCK</h2>
                <h4>Confidence: {conf*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(ROCK_IMG, width=150)
        else:
            st.markdown(f"""
            <div style="padding:20px; background-color:#FF5733; color:white; border-radius:10px; text-align:center;">
                <h2>Prediction: MINE</h2>
                <h4>Confidence: {conf*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(MINE_IMG, width=150)

def analytics_page():
    st.header("Model Analytics")
    if len(st.session_state["predictions"]) == 0:
        st.info("No predictions yet for analytics.")
        return

    df = pd.DataFrame(st.session_state["predictions"])
    st.subheader("Prediction Distribution")
    fig = px.pie(df, names="prediction", title="Rock vs Mine Distribution", color_discrete_map={"R":"green","M":"red"})
    st.plotly_chart(fig)

    st.subheader("Prediction Confidence Histogram")
    fig2 = px.histogram(df, x="confidence", nbins=20, title="Prediction Confidence")
    st.plotly_chart(fig2)

def history_page():
    st.header("Prediction History")
    if len(st.session_state["predictions"]) == 0:
        st.info("No predictions yet.")
        return

    df = pd.DataFrame(st.session_state["predictions"])
    st.dataframe(df[["prediction", "confidence"]])

    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv, file_name="prediction_history.csv")

# ------------------------------
# Main App
if not st.session_state["user_logged_in"]:
    login_page()
else:
    sidebar_metrics()  # show metrics on sidebar
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Home", "Predict", "Analytics", "History", "Logout"])

    if choice == "Home":
        home_page()
    elif choice == "Predict":
        predict_page()
    elif choice == "Analytics":
        analytics_page()
    elif choice == "History":
        history_page()
    elif choice == "Logout":
        st.session_state["user_logged_in"] = False
        st.experimental_rerun()
