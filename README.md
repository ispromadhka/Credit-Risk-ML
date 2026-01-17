# Credit Risk ML

Pet-project: CatBoost vs TurboCat comparison on credit risk prediction task.

## Quick Start (Docker)

```bash
docker-compose up --build
```

Open http://localhost:8888 in browser.

## Manual Installation

```bash
pip install -r requirements.txt
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

## Project Structure

```
.
├── data/                    # Dataset
├── notebooks/               # Jupyter notebooks
├── reports/                 # Generated plots
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Results

| Model    | ROC-AUC | F1-Score |
|----------|---------|----------|
| CatBoost | ~0.95   | ~0.72    |
| TurboCat | ~0.95   | ~0.70    |
