import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load model and components
model_path = "./model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Emoji map
emoji_dict = {
    "happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐",
    "fear": "😨", "disgust": "🤢", "surprise": "😲", "joy": "😄",
    "frustrated": "😤", "embarrassed": "😳", "excited": "🤩", "anxious": "😟",
}

# Session log init
if "log_df" not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=["Timestamp", "Text", "Emotion", "Emoji"])

def predict_emotion_with_probs(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class_id = np.argmax(probs)
    predicted_emotion = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_emotion, probs

# UI config
st.set_page_config(page_title="Emotion Analyzer", page_icon="🎭", layout="wide")
st.title("🎭 Emotion Analyzer")
st.caption("AI-powered emotion detection with instant visuals")

# Layout columns
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("✍️ What's on your mind?")
    user_input = st.text_area("Type something...", height=120, placeholder="e.g., I feel great today!")

with col2:
    st.markdown("### ")
    if st.button("🔍 Analyze"):
        if user_input.strip():
            emotion, probs = predict_emotion_with_probs(user_input)
            emoji = emoji_dict.get(emotion.lower(), "")
            st.success(f"**Emotion:** `{emotion}` {emoji}")

            new_row = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Text": user_input,
                "Emotion": emotion,
                "Emoji": emoji
            }])
            st.session_state.log_df = pd.concat([st.session_state.log_df, new_row], ignore_index=True)

            emotions = label_encoder.classes_
            df = pd.DataFrame({'Emotion': emotions, 'Probability': probs}).sort_values(by='Probability', ascending=False)

            # Probability bar chart
            st.subheader("📊 Prediction Confidence")
            st.bar_chart(df.set_index("Emotion"))

            # Pie chart
            st.subheader("🥧 Emotion Distribution")
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            colors = sns.color_palette("pastel")[0:len(df)]
            wedges, texts, autotexts = ax.pie(
                df["Probability"], labels=df["Emotion"],
                autopct="%1.1f%%", startangle=140,
                colors=colors, labeldistance=1.15
            )
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(8)
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please enter something to analyze.")

# CSV Upload
with st.expander("📁 Upload CSV for Bulk Emotion Analysis"):
    uploaded_file = st.file_uploader("Upload a CSV with a 'Text' column", type=["csv"])
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if 'Text' not in batch_df.columns:
                st.error("CSV must contain a 'Text' column.")
            else:
                results = []
                for text in batch_df['Text']:
                    emotion, _ = predict_emotion_with_probs(str(text))
                    emoji = emoji_dict.get(emotion.lower(), "")
                    results.append({"Text": text, "Emotion": emotion, "Emoji": emoji})
                output_df = pd.DataFrame(results)
                st.dataframe(output_df)

                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Result CSV", csv, "emotion_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# Session Log
with st.expander("📈 View This Session's Emotion Log"):
    if not st.session_state.log_df.empty:
        st.dataframe(st.session_state.log_df)

        trend_df = st.session_state.log_df["Emotion"].value_counts().reset_index()
        trend_df.columns = ["Emotion", "Count"]

        st.subheader("📊 Mood Trend")
        st.bar_chart(trend_df.set_index("Emotion"))

        log_csv = st.session_state.log_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Log", log_csv, "emotion_log.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("🚀 Powered by DistilBERT · PyTorch · Streamlit")
