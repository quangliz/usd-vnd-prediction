# USD/VND Exchange Rate Prediction

A machine learning project for predicting USD/VND exchange rates using XGBoost regression. The project consists of a FastAPI backend service and a Streamlit frontend dashboard for visualization and predictions.

## Features

- **ML Model**: XGBoost regression model trained on historical USD/VND exchange rate data
- **REST API**: FastAPI service with rate limiting for prediction endpoints
- **Interactive Dashboard**: Streamlit web app for visualizing historical data and predictions
- **S3 Integration**: Data and model storage on AWS S3
- **Feature Engineering**: Technical indicators including RSI, moving averages, lag features, and time-based features
- **Docker Support**: Containerized deployment for both API and Streamlit services

## Architecture

The project is split into two services:

1. **API Service** (`src/api.py`): FastAPI backend that serves predictions
   - Loads model and data from S3 on startup
   - Rate-limited prediction endpoint (5 requests per minute)
   - Health check endpoint

2. **Streamlit Service** (`main.py`): Frontend dashboard
   - Visualizes last 6 months of historical data
   - Interactive prediction interface (1-100 days ahead)
   - Real-time chart updates with Plotly

## Project Structure

```
usd-vnd-prediction/
├── src/
│   ├── api.py                 # FastAPI application
│   ├── inference_pipeline.py  # Prediction logic
│   ├── training_pipeline.py   # Model training, evaluation, tuning
│   ├── feature_pipeline.py    # Feature engineering
│   └── utils.py               # S3 utilities for data/model loading
├── notebooks/                 # Jupyter notebooks for EDA and experimentation
├── models/                    # Local model storage
├── main.py                    # Streamlit application
├── Dockerfile                 # API service container
├── Dockerfile.streamlit       # Streamlit service container
└── pyproject.toml             # Project dependencies with optional groups
```

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured (for S3 access)
- Docker (optional, for containerized deployment)

## Setup

### 1. Install Dependencies

The project uses optional dependency groups to minimize installation size:

```bash
# Install API dependencies only
uv sync --extra api

# Install Streamlit dependencies only
uv sync --extra streamlit

# Install all dependencies (for development/training)
uv sync --extra api --extra streamlit --extra training
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
# AWS Configuration
S3_BUCKET=your-bucket-name
REGION=your-aws-region
DATA_PATH_S3=path/to/data.csv
MODEL_PATH_S3=path/to/model.pkl

# AWS Credentials (or use AWS CLI/instance profile)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### 3. Run Locally

**Start API Service:**
```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**Start Streamlit Dashboard:**
```bash
uv run streamlit run main.py --server.port 8501
```

The Streamlit app will be available at `http://localhost:8501` and will connect to the API at `http://localhost:8000`.

## Docker Deployment

### Build Images

```bash
# Build API image
docker build -f Dockerfile -t usd-vnd-api .

# Build Streamlit image
docker build -f Dockerfile.streamlit -t usd-vnd-streamlit .
```

### Run Containers

```bash
# Run API service
docker run -d \
  --name usd-vnd-api \
  -p 8000:8000 \
  --env-file .env \
  usd-vnd-api

# Run Streamlit service
docker run -d \
  --name usd-vnd-streamlit \
  -p 8501:8501 \
  --env-file .env \
  usd-vnd-streamlit
```

**Note**: If running in Docker, update `API_URL` in `main.py` to point to your API container (e.g., `http://usd-vnd-api:8000` or use Docker networking).

### Docker Compose (Recommended)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    env_file:
      - .env
    depends_on:
      - api
    restart: unless-stopped
```

Then run:
```bash
docker-compose up -d
```

## API Endpoints

### Health Check
```bash
GET /health
```
Returns: `{"status": "healthy"}`

### Root
```bash
GET /
```
Returns: `{"message": "Hello, World!"}`

### Predict
```bash
GET /predict/{n}
```
Predicts the next `n` days of USD/VND exchange rates.

**Parameters:**
- `n` (path parameter): Number of days to predict (1-100)

**Response:**
```json
[
  {
    "Ngày": "2024-01-15",
    "Lần cuối": 24500.0
  },
  ...
]
```

**Rate Limit:** 5 requests per 60 seconds (returns 429 if exceeded)

## Model Training

### Train Model
```bash
uv run python -m src.training_pipeline train \
  --data-path path/to/data.csv \
  --save-model-path path/to/save/model
```

### Evaluate Model
```bash
uv run python -m src.training_pipeline eval \
  --data-path path/to/data.csv \
  --model-path path/to/model.pkl
```

### Hyperparameter Tuning
```bash
uv run python -m src.training_pipeline tune \
  --data-path path/to/data.csv \
  --n-trials 50 \
  --experiment-name xgboost_regression
```

Training uses MLflow for experiment tracking and Optuna for hyperparameter optimization.

## Features Used

The model uses the following engineered features:

- **Lag Features**: 1-day, 3-day, and 7-day lagged closing prices
- **Moving Averages**: 7-day and 30-day moving averages
- **Volatility**: 7-day standard deviation
- **Technical Indicators**: RSI (14-period)
- **Time Features**: Day of week, month, quarter, year
