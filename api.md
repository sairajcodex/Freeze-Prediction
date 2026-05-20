# Bank Risk API Documentation

This document describes the API endpoints provided by the Bank Risk Flask application. These endpoints are available for processing transaction and KYC (Know Your Customer) data to calculate risk scores.

## Base URL

If running locally typically: `http://localhost:5000` or `http://127.0.0.1:5000`. Replace with your deployed AWS/Render URL when using the API in a production environment.

---

## 1. Predict Risk (JSON API)

Calculates the transaction risk, KYC risk, and total freeze risk based on raw feature inputs. Does not perform image optical character recognition (OCR); instead, relies on pre-extracted KYC variables.

**URL:** `/predict_freeze_risk`  
**Method:** `POST`  
**Content-Type:** `application/json`

### Request Body

Requires a JSON object including both transaction and KYC features.

| Field | Type | Description |
|---|---|---|
| `transaction_amount` | Float | The transaction amount. |
| `income` | Float | The user's income. |
| `txn_income_ratio` | Float | Ratio of transaction amount to income. |
| `amount_user_mean_ratio` | Float | Ratio comparing the transaction to the user's mean transaction volume. |
| `rolling_5_txn_amount_std` | Float | Standard deviation of the previous 5 transactions. |
| `txn_count_last_10_min` | Float | The number of transactions the user initiated in the last 10 minutes. |
| `night_txn_ratio_7day` | Float | Ratio of user transactions that occur at night (over a 7-day window). |
| `name_mismatch_flag` | Float | Flag indicating mismatch between registered name and KYC doc name (0 or 1). |
| `invalid_id_format_flag` | Float | Flag indicating whether the id format is invalid (0 or 1). |
| `ocr_confidence` | Float | OCR reading confidence percentage/ratio. |

#### Example Request

```json
{
  "transaction_amount": 1500.0,
  "income": 5000.0,
  "txn_income_ratio": 0.3,
  "amount_user_mean_ratio": 1.2,
  "rolling_5_txn_amount_std": 200.5,
  "txn_count_last_10_min": 1.0,
  "night_txn_ratio_7day": 0.05,
  "name_mismatch_flag": 0.0,
  "invalid_id_format_flag": 0.0,
  "ocr_confidence": 0.98
}
```

### Response

Returns a JSON object with calculated risk scores between `0` and `1`.

| Field | Type | Description |
|---|---|---|
| `transaction_risk_score` | Float | Output from Stage 1: The Transaction Sub-Model risk. |
| `kyc_risk_score` | Float | Output from Stage 2: The KYC Sub-Model risk. |
| `freeze_risk_score` | Float | Output from Stage 3: The Ensembled Fusion Model risk. |

#### Example Response

```json
{
  "transaction_risk_score": 0.2312,
  "kyc_risk_score": 0.0451,
  "freeze_risk_score": 0.1349
}
```

---

## 2. Predict UI Pipeline (Multipart Form-Data API)

Designed for front-end integrations with file uploads. Handles complete end-to-end processing, including reading the uploaded KYC image via Computer Vision, extracting features dynamically, and processing risk scores.

**URL:** `/predict_ui`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

### Request Payload

Submit via Form-Data containing file upload and text values.

| Key | Type | Description |
|---|---|---|
| `kyc_image` | File | The user's KYC document image (JPEG, PNG, etc). |
| `registered_name` | String | The user's registered name against which to compare the document. |
| `transaction_amount` | Float | The transaction amount. |
| `income` | Float | The user's income. |
| `txn_income_ratio` | Float | Ratio of transaction amount to income. |
| `amount_user_mean_ratio` | Float | Ratio comparing the transaction to the user's mean transaction volume. |
| `rolling_5_txn_amount_std` | Float | Standard deviation of the last 5 transactions. |
| `txn_count_last_10_min` | Float | Transactions initiated within the last 10 minutes. |
| `night_txn_ratio_7day` | Float | Ratio of nocturnal transactions over 7 days. |

### Response

Returns a JSON object combining final risk scores with the newly computed OpenCV data points.

#### Example Response

```json
{
  "transaction_risk_score": 0.7511,
  "kyc_risk_score": 0.8223,
  "freeze_risk_score": 0.8105,
  "name_mismatch_flag": 1.0,
  "invalid_id_format_flag": 0.0,
  "ocr_confidence": 0.85,
  "extracted_text": "DOE JOHN..."
}
```

*(Note: `extracted_text` and any other dynamic features depend on `cv.py`'s exact output schema).*

---

## 3. Web UI 

**URL:** `/`  
**Method:** `GET`  

Serve the main HTML web application dashboard from `templates/index.html`. Normally accessed via a web browser rather than programmably.

```
