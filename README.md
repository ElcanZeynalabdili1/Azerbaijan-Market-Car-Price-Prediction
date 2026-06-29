# 🚗 Azerbaijan Market Car Price Prediction

A machine learning web application that predicts used car prices based on ~31,000 listings scraped from [turbo.az](https://turbo.az) in 2022.

---

## 🖥️ Demo

> Run locally with Streamlit — enter car details and get an instant price prediction in AZN, USD, and EUR.

---

## 📌 Features

- Predicts car prices based on 11 features (brand, model, age, engine, mileage, etc.)
- Multi-currency output: AZN, USD, EUR
- Clean and interactive Streamlit UI
- Trained on real Azerbaijani market data (~31,000 listings)

---

## 🧠 ML Pipeline

| Step | Details |
|------|---------|
| Data | ~31,000 turbo.az listings (2022) |
| Target | Log-transformed price (AZN) |
| Features | Brand, model, car age, body type, color, engine volume, engine power, gearbox, fuel type, log-mileage, city |
| Models tested | HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor |
| Final model | Best performing model saved as `car_price_model1.pkl` |

---

## 📁 Project Structure

```
├── app.py                              # Streamlit web application
├── Car_Price_Prediction_Turbo_az_data.ipynb  # Data analysis & model training
├── car_price_model.pkl                 # Trained ML model (v1)
├── car_price_model1.pkl                # Trained ML model (v2 - production)
├── turbo-az-2022- 1-ci-hisse.csv      # Raw data - part 1
├── turbo-az-2022-2 ci-hisse.csv       # Raw data - part 2
├── turbo-az-2022-3 cu hisse.csv       # Raw data - part 3
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/ElcanZeynalabdili1/Azerbaijan-Market-Car-Price-Prediction.git
cd Azerbaijan-Market-Car-Price-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas & NumPy** — data processing
- **Scikit-learn** — model training
- **Streamlit** — web interface
- **Joblib** — model serialization

---

## 📊 Input Features

| Feature | Description |
|---------|-------------|
| Marka | Car brand (e.g. BMW, Toyota) |
| Model | Car model (e.g. 5 Series) |
| Buraxılış ili | Year of manufacture |
| Ban növü | Body type (Sedan, SUV, etc.) |
| Sürət qutusu | Gearbox type |
| Yanacaq növü | Fuel type |
| Rəng | Color |
| Mühərrik həcmi | Engine volume (L) |
| Mühərrik gücü | Engine power (hp) |
| Yürüş | Mileage (km) |
| Şəhər | City |

---

## 👤 Author

**Elcan Zeynalabdili**  
[GitHub](https://github.com/ElcanZeynalabdili1)

---

## 📄 License

This project is licensed under the MIT License.
