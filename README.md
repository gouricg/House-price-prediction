# California House Price Prediction — Machine Learning Project

**B.Tech Computer Science & Engineering | Machine Learning Assignment**  
**Course Outcome (CO5):** *Solve real-life problems using appropriate machine learning models and evaluate performance measures.*

---

## 📌 Repository Structure

```
.
├── app.py              # Flask REST API server for real-time model inference
├── model.pkl           # Serialized Gradient Boosting Regressor (Scikit-Learn Pipeline)
├── dataset.csv         # California Housing dataset (20,640 records)
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation and quickstart guide
├── report.pdf          # Full academic B.Tech project report
└── screenshots/
    └── app.png         # Mobile application UI interface screenshot
```

---

## 📊 Model Evaluation Results

- **Selected Model:** Gradient Boosting Regressor
- **5-Fold Cross-Validation RMSE:** `0.4808`
- **Test Mean Absolute Error (MAE):** `0.3209` (~$32,089 USD)
- **Test Root Mean Squared Error (RMSE):** `0.4808` (~$48,079 USD)
- **Coefficient of Determination ($R^2$):** `0.8236` (**82.36%** variance explained)

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python app.py
```
*Server runs on `http://localhost:5000`.*

### 3. Test Prediction Endpoint
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 6.2,
    "HouseAge": 20.0,
    "AveRooms": 6.5,
    "AveBedrms": 3.0,
    "Population": 1200.0,
    "AveOccup": 3.4,
    "Latitude": 34.2,
    "Longitude": -118.4
  }'
```

**Response:**
```json
{
  "formatted_price": "$278,451.17",
  "model_name": "Gradient Boosting Regressor",
  "predicted_price_usd": 278451.17,
  "predicted_value": 2.7845,
  "success": true,
  "target_unit": "100,000 US dollars"
}
```
