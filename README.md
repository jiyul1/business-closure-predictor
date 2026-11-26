# Industry Closure Rate Prediction Web App

This Streamlit app uses three Orange Random Forest models to estimate industry-level closure rates one, two, and three years ahead, using the 2025 Seoul industry dataset.

## Files
- `app.py`: Streamlit application
- `data/industry_data.csv`: input data
- `models/model_1y.pkcls`: one-year model
- `models/model_2y.pkcls`: two-year model
- `models/model_3y.pkcls`: three-year model
- `orange_workflow.ows`: original Orange workflow

## Environment
- Python 3.10
- Package versions are pinned in `requirements.txt`.
- The models were saved with scikit-learn 1.1.3.

## Deploy to Streamlit Community Cloud
1. Upload all files and folders in this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Select Python 3.10 in Advanced settings.
5. Deploy.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Important limitations
The result is an estimate of an industry-level closure rate, not the probability that an individual new store will close. Predictions depend on the historical data and model quality. The 2025 one-year target may have been included in training, so its prediction may not represent independent future validation.
