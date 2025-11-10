# Short-Term Energy Load Forecasting: A Dual-Model Approach (TSFM & LightGBM)

This project implements a dual-model solution for short-term energy load forecasting in commercial buildings. It combines a traditional machine learning model (LightGBM) trained on engineered time-series features with a foundation model approach using IBM's Tiny Time Mixer (TSFM) for zero-shot forecasting. Both models operate on a shared preprocessing pipeline to enable direct, meaningful comparison.

---

## Methodology

### 1. Data Preprocessing and Feature Engineering

The raw time-series data is transformed into a structured format to capture temporal dependencies and seasonal behavior.

**Time-Based Features**
- `hour`
- `dayofweek`
- `month`
- `day`
- `week`

**Lag Features**
To incorporate temporal context and trends:
- `lag_1`: Value from 1 hour prior (recent trend)
- `lag_24`: Value from 24 hours prior (daily seasonality)
- `lag_168`: Value from 168 hours prior (weekly seasonality)

**Scaling**
- Time-based features are normalized using `StandardScaler`.

**Outputs**
- `train_features.csv`
- `test_features.csv`

---

### 2. Model 1: LightGBM (Gradient Boosting)

A classical supervised regression approach using the LightGBM library.

**Configuration**
- Type: Gradient Boosting Decision Tree Regressor (`gbdt`)
- Training: 80% training, 20% validation split
- Optimization: Early stopping (50 rounds) on validation loss

**Prediction Output**
- `submissionLightGBM.csv`

**Performance**
- Validation RMSE: **0.8999**  
(Computed on the validation split of `train_features.csv`)

---

### 3. Model 2: Tiny Time Mixer (TSFM) - Zero-Shot Forecasting

A Time-Series Foundation Model used without fine-tuning.

**Model**
- `ibm-granite/granite-timeseries-ttm-r1` (via Hugging Face)

**Data Preparation**
- The raw `test.csv` is transformed into TSFM-compatible context windows of length 512 using `TimeSeriesPreprocessor`.

**Forecasting**
- A `TimeSeriesForecastingPipeline` generates 96-step predictions.
- The required 24-hour prediction window is extracted for submission.

**Prediction Output**
- `submissionTSFM.csv`

---

## How to Run

1. **Data**
   Ensure the competition dataset is available in your environment:
`train.csv`
`test.csv`
`sample_submission.csv`


2. **Dependencies**
Run the installation commands provided in the notebook to set up:
- `granite-tsfm`
- `transformers`
- `lightgbm`
- Standard Python scientific libraries

3. **Execution**
Run all notebook cells in order. The preprocessing pipeline will run once and serve both models.

4. **Outputs**
After execution, final prediction files will be created in:
```bash
/kaggle/working/
├── submissionLightGBM.csv
└── submissionTSFM.csv
```

---

## Repository Contents (if applicable)

- Notebook: `short-term-energy-forecast-using-tsfm-and-lgbm.ipynb`
- Processed data: `test.csv`, (`train.csv` is too big for GitHub)
- Final predictions: `lightgbm.csv`, `tsfm.csv`

---

This setup allows for a clear comparison between foundation-model zero-shot forecasting and traditional supervised learning methods, highlighting tradeoffs between model efficiency, data dependency, and predictive accuracy.
