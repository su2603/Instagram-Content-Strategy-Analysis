# 📊 Instagram Content Strategy Analyzer

This project provides a comprehensive data-driven approach to analyzing Instagram content performance, detecting patterns, modeling engagement, and offering optimization strategies.

---

## 📁 Module: `instagramContStratAnalysis.py`

### 🔧 Class: `InstagramAnalyzer`

#### **Purpose:**

Encapsulates the full data science pipeline for Instagram content analysis — from data preprocessing, feature extraction, and anomaly detection, to time series forecasting, clustering, causal inference, and model interpretation.

---

### 📥 1. Data Loading & Preprocessing

#### `load_and_preprocess_data()`

- **Input**: CSV file with Instagram posts
- **Steps**:
  - Converts 'Publish time' to datetime
  - Adds calendar features (`Hour`, `Day of Week`, `Month`)
  - Calculates `Total Engagement` as the sum of likes, comments, shares, etc.
  - Computes `Engagement Rate` (relative to followers if available)

---

### 🧠 2. Feature Engineering

#### `feature_engineering()`

- **Text-Based Features**: `Description Length`, `Word Count`, `Hashtag Count`, `Emoji Count`, etc.
- **Sentiment**: Polarity and subjectivity via TextBlob
- **Temporal Cycles**: Adds sin/cos transforms for hour/day/month
- **Rolling Stats**: Calculates rolling mean and std for engagement metrics
- **Historical Features**: Previous post and average of last 3 posts
- **Post Timing**: Peak hour, business hour, evening indicator

---

### 🚨 3. Anomaly & Outlier Detection

#### `detect_anomalies_and_outliers()`

- Uses `IsolationForest` to identify unusual patterns
- Flags statistical outliers with Z-scores
- Marks top 5% engagement posts as `Is_Viral`

---

### ⏱️ 4. Time Series Analysis & Forecasting

#### `perform_time_series_analysis()`

- Resamples engagement data by day
- Computes 7-day rolling mean (`Engagement_Trend`) and volatility
- Adds seasonal indicators and fits linear regression with time + day-of-week sin/cos
- Forecasts future engagement and visualizes trends + confidence intervals

---

### 🎯 5. Causal Inference

#### `perform_causal_inference()`

- Evaluates treatment effects of:
  - Call to Action (CTA)
  - Weekend posting
  - Peak Hour
- Calculates **Average Treatment Effect (ATE)** and p-values
- Visualizes statistically significant drivers of engagement

---

### 🧪 6. A/B Test Simulation

#### `simulate_ab_tests()`

- Simulates experiments like:
  - Long vs short descriptions
  - High vs low hashtag count
  - Peak vs off-hours
- Computes effect size, confidence intervals, and recommendations

---

### 📊 7. Clustering & Content Segmentation

#### `perform_clustering_analysis()`

- Applies KMeans to engineered features
- Selects optimal clusters via elbow method
- Visualizes clusters using PCA
- Useful to identify high-performing content patterns

---

### 📈 8. Correlation Network

#### `perform_correlation_network_analysis()`

- Heatmap of correlations between numeric features
- Highlights top features linked to `Total Engagement`

---

### 🤖 9. Machine Learning Modeling

#### `train_multiple_models(X, y, feature_names)`

- Trains multiple regressors:
  - RandomForest, XGBoost, Ridge, SVR, MLP, Gaussian Process
- Uses `GridSearchCV` for hyperparameter tuning
- Stores metrics: RMSE, R², MAE
- Extracts feature importances for tree-based models

---

### 🧬 10. Ensemble Learning

#### `create_ensemble_model(X_train, y_train)`

- Builds `VotingRegressor` using top 3 models by R²
- Combines predictions to improve stability

---

### 🔍 11. Model Explainability

#### `model_interpretability()`

- Uses permutation importance (like SHAP)
- Visualizes top driver features
- Plots **partial dependence** of top 5 features to show their marginal effects

---

### 🧪 12. Feature Selection

#### `perform_automated_feature_selection(X, y)`

Applies:

- `SelectKBest` (univariate regression)
- `RFECV` (Recursive Feature Elimination with Cross-Validation)
- `Lasso` regularization
- Returns consensus features

---

### 💡 13. Strategic Insights

#### `generate_insights_and_recommendations()`

Prints:

- Best-performing content cluster
- Top posting hours and days
- Description length vs engagement
- Optimal hashtags

---

### 📊 14. Interactive Dashboard

#### `create_interactive_dashboard()`

Generates 2×2 Plotly dashboard showing:

- Engagement trends
- Post type effectiveness
- Hourly insights
- Feature importances

---

## 📁 App: `instaDataAnalysisApp.py`

### 🔍 Purpose:

Streamlit-based interface for exploring and analyzing Instagram content data interactively.

### Key Features:

- Upload or simulate Instagram data
- Visual breakdowns of:
  - Engagement by hour/day
  - Cluster performance
  - Time series trends
  - Predictive modeling (RandomForest)
  - Viral post characteristics

---

## 🛠 Technical Dependencies

- `scikit-learn`: modeling, feature selection, scaling
- `xgboost`: advanced boosting models
- `TextBlob`: NLP sentiment
- `matplotlib`, `seaborn`, `plotly`: visualizations
- `pandas`, `numpy`: data wrangling
- `streamlit`: UI

---
