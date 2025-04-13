# Emotion-Analyzer
Emotion Analyzer – A Streamlit app that uses a fine-tuned DistilBERT model to detect emotions from text with visual insights.
Here's a well-structured `README.md` file for your **Emotion Analyzer** project, which includes details about the app, setup instructions, training, and usage:

Emotion Analyzer is an AI-powered web application that detects human emotions from text using a fine-tuned DistilBERT model. It provides real-time predictions with emoji visuals and probability charts. You can also analyze emotions in bulk from CSV files.

🧠 Model Info

- Base Model: `distilbert-base-uncased`
- Task: Emotion Classification
- Training Dataset: MELD-style CSVs (`train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`)
- Frameworks: PyTorch, Hugging Face Transformers, Streamlit

---
 🚀 Features

- Predict emotions from typed text with emoji indicators.
- View confidence bar and pie charts.
- Upload CSV for batch emotion analysis.
- Log session predictions and download logs.
- Clean UI with expandable sections for bulk analysis and trends.

---

## 📁 Project Structure

```
├── app.py                   # Streamlit app
├── train.py                 # Model training script
├── train_sent_emo.csv       # Training dataset
├── dev_sent_emo.csv         # Validation dataset
├── test_sent_emo.csv        # Test dataset
├── model/                   # Saved model and tokenizer (after training)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── label_encoder.pkl
```

---

⚙️ Setup Instructions

 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotion-analyzer.git
cd emotion-analyzer
```

 2. Install Dependencies
```bash
pip install -r requirements.txt
```

<details>
```
streamlit
torch
transformers
pandas
scikit-learn
joblib
matplotlib
seaborn
tqdm
```
</details>

 3. (Optional) Train the Model
Run the training script if you want to train or fine-tune the model:
```bash
python train.py
```

This will save the model and `label_encoder.pkl` to the `./model` folder.

---

🧪 Run the App

```bash
streamlit run app.py
```

---

## 🧬 Input Format

 For Single Prediction
Just type into the textbox on the app.
 For Batch Prediction
Upload a CSV with the following structure:

```csv
Text
I am feeling great today!
This is really upsetting.
I’m so excited for the trip!
```

---

## 📤 Output

- **Single input:** Displays predicted emotion with emoji, confidence bar chart, and pie chart.
- **Batch input:** Returns a downloadable CSV with `Text`, `Emotion`, and `Emoji` columns.
- **Log file:** Tracks your session interactions and can be downloaded.

---


## 📜 License

This project is licensed under the MIT License.

---
