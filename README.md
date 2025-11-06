# 🍷 Wine Quality Prediction

A Streamlit app that predicts red wine quality using a trained Random Forest Regressor on the UCI Wine Quality dataset.

## Project Structure

```
wine-quality-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── wine_model.pkl        # generated after training
├── scaler.pkl            # generated after training
└── README.md
```

## Local Development

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (generates `wine_model.pkl` and `scaler.pkl`):

```bash
python train_model.py
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

- Push this project to a GitHub repository
- On Streamlit Cloud, select the repo and set main file to `app.py`
- Deploy

## Features
- Interactive sliders for all input features
- Real-time quality prediction with gauge meter
- Data exploration with Plotly charts
- Model info and feature importance
- Responsive UI with tabs

## Dataset
- UCI ML Repository: Red Wine Quality

---
Built with ❤️ using Streamlit and scikit-learn.



