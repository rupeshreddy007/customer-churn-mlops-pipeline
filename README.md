# Customer Churn MLOps Pipeline

A production-ready MLOps pipeline for predicting customer churn using machine learning best practices, data versioning, and experiment tracking.

##  Project Overview

This project implements an end-to-end machine learning pipeline for customer churn prediction with:

- **Data Management**: DVC (Data Version Control) for managing datasets
- **Experiment Tracking**: MLflow for tracking models and metrics
- **Pipeline Orchestration**: DVC pipelines for reproducible workflows
- **API Serving**: FastAPI for model inference endpoints
- **Containerization**: Docker support for deployment
- **ML Models**: scikit-learn based classification models

##  Project Structure

```
├── src/                      # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature engineering
│   ├── models/              # Model training and evaluation
│   └── pipeline/            # Pipeline utilities
├── training/                # Training scripts
│   └── train.py            # Main training pipeline
├── inference/               # Inference API
│   ├── app.py              # FastAPI application
│   └── schema.py           # Request/response schemas
├── data/                    # Data storage (tracked by DVC)
│   ├── raw/                # Raw data
│   └── processed/          # Processed data
├── models/                  # Trained models (tracked by DVC)
├── reports/                 # Generated reports and metrics
├── dvc.yaml                 # DVC pipeline configuration
├── params.yaml              # Parameters
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
└── README.md               # This file
```

##  Quick Start

### Prerequisites

- Python 3.8+
- Git
- DVC
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/rupeshreddy007/customer-churn-mlops-pipeline.git
cd customer-churn-mlops-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize DVC**
```bash
dvc init
dvc pull  # If using remote storage
```

##  Training

### Run the Training Pipeline

The training pipeline is orchestrated using DVC:

```bash
dvc repro
```

This will execute:
- Data loading and preprocessing
- Feature engineering
- Model training
- Model evaluation
- Metrics generation

### Monitor with MLflow

```bash
mlflow ui
```

Then visit `http://localhost:5000` to view:
- Experiment runs
- Model metrics
- Parameter configurations
- Model artifacts

##  Inference

### Start the API Server

```bash
python -m uvicorn inference.app:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- **POST /predict**: Make predictions on customer churn
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345",
    "monthly_charges": 65.5,
    "tenure": 12,
    ...
  }'
```

- **GET /health**: Health check
```bash
curl http://localhost:8000/health
```

##  Docker

### Build Docker Image

```bash
docker build -t customer-churn-mlops:latest .
```

### Run Container

```bash
docker run -p 8000:8000 customer-churn-mlops:latest
```

##  Development

### Code Structure

- **Data Pipeline** (`src/data/`): Handles data loading and preprocessing
- **Features** (`src/features/`): Feature engineering and transformation
- **Models** (`src/models/`): Model training, evaluation, and utilities
- **Utils** (`src/pipeline/`): Common utilities and helpers

### Updating Parameters

Edit `params.yaml` to adjust:
- Model hyperparameters
- Data preprocessing settings
- Feature engineering options
- Train/test split ratios

##  Metrics

The pipeline generates metrics in `reports/metrics.json`:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

##  Data Versioning

All datasets and models are versioned using DVC:

```bash
# Pull remote data
dvc pull

# Push updated data
dvc push

# View data history
dvc dag
```

##  Contributing

Contributions are welcome! Please:

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Push and create a pull request

##  License

This project is licensed under the MIT License.

##  Contact

For questions or issues, please create an issue in the repository.

---

**Built with**: MLflow • DVC • FastAPI • scikit-learn • pandas • Docker
