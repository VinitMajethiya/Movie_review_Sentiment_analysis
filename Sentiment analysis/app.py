import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------- Constants --------------------
MAX_LEN = 200
TOKENIZER_PATH = "tokenizer.pkl"
MODEL_PATH = "movie_review_model.keras"

# -------------------- Load Assets --------------------
@st.cache_resource
def load_tokenizer():
    return joblib.load(TOKENIZER_PATH)

@st.cache_resource
def load_sentiment_model():
    return load_model(MODEL_PATH)

tokenizer = load_tokenizer()
model = load_sentiment_model()

# -------------------- Streamlit UI Config --------------------
st.set_page_config(
    page_title="🎥 Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with black background and black font for text area
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: #000000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    .stTextArea textarea, .stTextInput input {
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 12px;
        background-color: #ffffff;
        color: #000000 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTextArea textarea::placeholder {
        color: #6b7280;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stDownloadButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stDownloadButton>button:hover {
        background-color: #059669;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stAlert {
        border-radius: 8px;
        padding: 12px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    .sidebar .stButton>button {
        width: 100%;
        margin-bottom: 10px;
    }
    .review-container {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        margin-top: 20px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("🎬 Sentiment Analyzer")
    st.markdown("**Welcome to the Movie Review Sentiment Analyzer!**")
    st.markdown("Use this tool to predict whether movie reviews are positive or negative using a deep learning model.")
    st.markdown("---")
    st.subheader("Options")
    show_batch = st.button("📁 Upload Batch File", help="Click to upload a CSV or Excel file for batch predictions")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Built for a college project using Streamlit and TensorFlow. Analyzes sentiments with high accuracy!")

# -------------------- Main Content --------------------
st.title("🎥 Movie Review Sentiment Analyzer")
st.markdown("Discover the sentiment behind movie reviews with our AI-powered tool. Enter a single review or upload a file for batch analysis.")

# -------------------- Single Review Input --------------------
st.subheader("✍️ Analyze a Single Review")
with st.container():
    st.markdown('<div class="review-container">', unsafe_allow_html=True)
    with st.form("single_review_form"):
        user_review = st.text_area(
            "Enter your movie review",
            placeholder="e.g., 'This movie was absolutely fantastic with great acting!'",
            height=120,
            help="Type a movie review (at least a few sentences) to analyze its sentiment."
        )
        submitted = st.form_submit_button("🔍 Predict Sentiment")

    if submitted:
        if user_review.strip() == "":
            st.warning("⚠️ Please enter a review before predicting!")
        else:
            with st.spinner("Analyzing sentiment..."):
                seq = tokenizer.texts_to_sequences([user_review])
                padded = pad_sequences(seq, maxlen=MAX_LEN)
                pred = model.predict(padded)[0][0]

                st.markdown(f"**Prediction Score:** `{pred:.4f}`")
                if pred >= 0.5:
                    st.success("🟢 **Positive Sentiment** - This review expresses positive feelings!")
                else:
                    st.error("🔴 **Negative Sentiment** - This review expresses negative feelings!")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Batch Upload Section --------------------
if show_batch:
    st.subheader("📂 Batch Review Analysis")
    with st.container():
        st.markdown('<div class="review-container">', unsafe_allow_html=True)
        with st.expander("Upload a CSV or Excel File", expanded=True):
            st.markdown("**Instructions**: Upload a file with a 'review' column containing movie reviews.")
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=["csv", "xlsx"],
                help="The file must have a column named 'review' with movie reviews."
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    if 'review' not in df.columns:
                        st.error("❌ The file must contain a column named 'review'.")
                    else:
                        with st.spinner("Processing batch predictions..."):
                            sequences = tokenizer.texts_to_sequences(df['review'].astype(str))
                            padded = pad_sequences(sequences, maxlen=MAX_LEN)
                            predictions = model.predict(padded).flatten()

                            df['prediction_score'] = predictions
                            df['sentiment'] = np.where(predictions >= 0.5, '🟢 Positive', '🔴 Negative')

                            st.success("✅ Batch predictions completed!")
                            st.dataframe(
                                df[['review', 'sentiment', 'prediction_score']],
                                use_container_width=True,
                                height=300
                            )

                            csv_data = df.to_csv(index=False).encode('utf-8')
                            buffer = BytesIO()
                            df.to_excel(buffer, index=False, engine='openpyxl')
                            excel_data = buffer.getvalue()

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📥 Download CSV",
                                    data=csv_data,
                                    file_name="sentiment_predictions.csv",
                                    mime="text/csv"
                                )
                            with col2:
                                st.download_button(
                                    "📥 Download Excel",
                                    data=excel_data,
                                    file_name="sentiment_predictions.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                except Exception as e:
                    st.error(f"⚠️ Error processing file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown('<div class="footer">Built with ❤️ using Streamlit | College Project 2025</div>', unsafe_allow_html=True)