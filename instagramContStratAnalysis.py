import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    IsolationForest,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    PolynomialFeatures,
    RobustScaler,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import xgboost as xgb
from textblob import TextBlob
import re
import json
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.inspection import permutation_importance
from sklearn.calibration import cross_val_predict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


class InstagramAnalyzer:
    def __init__(self, file_path):
        """Initialize the analyzer with data loading and preprocessing"""
        self.file_path = file_path
        self.data = None
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.time_series_data = None
        self.content_embeddings = None
        self.competitor_data = None
        self.causal_insights = {}
        self.ab_test_results = {}
        self.forecast_model = None

    def load_and_preprocess_data(self):
        """Load and perform initial preprocessing"""
        print("Loading and preprocessing data...")

        # Load the CSV file
        self.data = pd.read_csv(self.file_path)

        # Convert 'Publish time' to datetime
        self.data["Publish time"] = pd.to_datetime(self.data["Publish time"])

        # Basic datetime features
        self.data["Day of Week"] = self.data["Publish time"].dt.day_name()
        self.data["Hour"] = self.data["Publish time"].dt.hour
        self.data["Month"] = self.data["Publish time"].dt.month
        self.data["Day of Month"] = self.data["Publish time"].dt.day
        self.data["Quarter"] = self.data["Publish time"].dt.quarter
        self.data["Is Weekend"] = (
            self.data["Publish time"].dt.weekday.isin([5, 6]).astype(int)
        )

        # Calculate engagement metrics
        engagement_metrics = ["Likes", "Shares", "Comments", "Saves", "Follows"]
        self.data["Total Engagement"] = self.data[engagement_metrics].sum(axis=1)

        # Engagement rate (relative to followers if available)
        if "Followers" in self.data.columns:
            self.data["Engagement Rate"] = (
                self.data["Total Engagement"] / self.data["Followers"]
            )

        print(f"Data loaded successfully. Shape: {self.data.shape}")

    def feature_engineering(self):
        """Perform  feature engineering"""
        print("Performing feature engineering...")

        # 1. Text Analysis Features
        if "Description" in self.data.columns:
            # Basic text metrics
            self.data["Description Length"] = self.data["Description"].str.len()
            self.data["Word Count"] = self.data["Description"].str.split().str.len()
            self.data["Hashtag Count"] = self.data["Description"].str.count("#")
            self.data["Mention Count"] = self.data["Description"].str.count("@")
            self.data["URL Count"] = self.data["Description"].str.count(r"http[s]?://")
            self.data["Emoji Count"] = self.data["Description"].str.count(
                r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
            )

            # Sentiment analysis
            self.data["Sentiment"] = self.data["Description"].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
            )
            self.data["Subjectivity"] = self.data["Description"].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0
            )

            # Content type indicators
            self.data["Has Question"] = (
                self.data["Description"]
                .str.contains(r"\?", case=False, na=False)
                .astype(int)
            )
            self.data["Has Exclamation"] = (
                self.data["Description"]
                .str.contains(r"!", case=False, na=False)
                .astype(int)
            )
            self.data["Has Call to Action"] = (
                self.data["Description"]
                .str.contains(
                    r"\b(follow|like|share|comment|save|swipe|click|link|bio)\b",
                    case=False,
                    na=False,
                )
                .astype(int)
            )

        # 2. Temporal Features
        self.data["Hour Sin"] = np.sin(2 * np.pi * self.data["Hour"] / 24)
        self.data["Hour Cos"] = np.cos(2 * np.pi * self.data["Hour"] / 24)
        self.data["Day Sin"] = np.sin(
            2 * np.pi * self.data["Publish time"].dt.dayofweek / 7
        )
        self.data["Day Cos"] = np.cos(
            2 * np.pi * self.data["Publish time"].dt.dayofweek / 7
        )
        self.data["Month Sin"] = np.sin(2 * np.pi * self.data["Month"] / 12)
        self.data["Month Cos"] = np.cos(2 * np.pi * self.data["Month"] / 12)

        # 3. Engagement Distribution Features
        for metric in ["Likes", "Shares", "Comments", "Saves", "Follows"]:
            if metric in self.data.columns:
                # Rolling statistics (if we have enough data points)
                if len(self.data) > 10:
                    self.data[f"{metric} Rolling Mean"] = (
                        self.data[metric].rolling(window=5, min_periods=1).mean()
                    )
                    self.data[f"{metric} Rolling Std"] = (
                        self.data[metric]
                        .rolling(window=5, min_periods=1)
                        .std()
                        .fillna(0)
                    )

                # Engagement ratios
                if self.data["Total Engagement"].sum() > 0:
                    self.data[f"{metric} Ratio"] = (
                        self.data[metric] / self.data["Total Engagement"]
                    )

        # 4. Post timing features
        self.data["Is Peak Hour"] = (
            self.data["Hour"].isin([8, 9, 12, 17, 18, 19, 20, 21]).astype(int)
        )
        self.data["Is Business Hour"] = self.data["Hour"].between(9, 17).astype(int)
        self.data["Is Evening"] = self.data["Hour"].between(18, 22).astype(int)

        # 5. Historical performance features (if we have chronological data)
        self.data = self.data.sort_values("Publish time")
        self.data["Prev Post Engagement"] = (
            self.data["Total Engagement"].shift(1).fillna(0)
        )
        self.data["Avg Last 3 Posts"] = (
            self.data["Total Engagement"]
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)
            .fillna(0)
        )

        print("Feature engineering completed.")

    def detect_anomalies_and_outliers(self):
        """Detect anomalous posts and engagement outliers"""
        print("Detecting anomalies and outliers...")

        # Select numerical features for anomaly detection
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data = self.data[numeric_features].fillna(0)

        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = self.anomaly_detector.fit_predict(feature_data)

        self.data["Is_Anomaly"] = (anomaly_labels == -1).astype(int)

        # Statistical outlier detection for engagement
        engagement_z_scores = np.abs(stats.zscore(self.data["Total Engagement"]))
        self.data["Is_Outlier"] = (engagement_z_scores > 3).astype(int)

        # Viral content detection (top 5% engagement)
        engagement_threshold = self.data["Total Engagement"].quantile(0.95)
        self.data["Is_Viral"] = (
            self.data["Total Engagement"] >= engagement_threshold
        ).astype(int)

        print(f"Detected {self.data['Is_Anomaly'].sum()} anomalies")
        print(f"Detected {self.data['Is_Outlier'].sum()} statistical outliers")
        print(f"Identified {self.data['Is_Viral'].sum()} viral posts")

    def perform_time_series_analysis(self):
        """Perform time series analysis and forecasting"""
        print("Performing time series analysis...")

        # Create time series data
        ts_data = (
            self.data.set_index("Publish time")
            .resample("D")["Total Engagement"]
            .agg(["mean", "sum", "count"])
        )
        ts_data.columns = ["Avg_Engagement", "Total_Daily_Engagement", "Posts_Count"]
        ts_data = ts_data.fillna(0)

        self.time_series_data = ts_data

        # Trend analysis
        ts_data["Engagement_Trend"] = ts_data["Avg_Engagement"].rolling(window=7).mean()
        ts_data["Engagement_Volatility"] = (
            ts_data["Avg_Engagement"].rolling(window=7).std()
        )

        # Seasonality detection
        ts_data["Day_of_Week"] = ts_data.index.dayofweek
        ts_data["Month"] = ts_data.index.month

        # Growth rate calculation
        ts_data["Engagement_Growth"] = ts_data["Avg_Engagement"].pct_change().fillna(0)

        # Simple forecasting (Linear trend + seasonality)
        if len(ts_data) > 14:  # Need sufficient data for forecasting
            from sklearn.linear_model import LinearRegression

            # Prepare forecasting features
            ts_data["Time_Index"] = range(len(ts_data))
            ts_data["Sin_Day"] = np.sin(2 * np.pi * ts_data["Day_of_Week"] / 7)
            ts_data["Cos_Day"] = np.cos(2 * np.pi * ts_data["Day_of_Week"] / 7)

            # Train simple forecast model
            forecast_features = ["Time_Index", "Sin_Day", "Cos_Day"]
            X_ts = ts_data[forecast_features].fillna(0)
            y_ts = ts_data["Avg_Engagement"]

            self.forecast_model = LinearRegression().fit(X_ts, y_ts)

            # Generate 7-day forecast
            future_dates = pd.date_range(
                start=ts_data.index[-1] + timedelta(days=1), periods=7
            )
            future_features = pd.DataFrame(
                {
                    "Time_Index": range(len(ts_data), len(ts_data) + 7),
                    "Sin_Day": np.sin(2 * np.pi * future_dates.dayofweek / 7),
                    "Cos_Day": np.cos(2 * np.pi * future_dates.dayofweek / 7),
                }
            )

            forecast = self.forecast_model.predict(future_features)

            # Visualize time series with forecast
            plt.figure(figsize=(15, 8))
            plt.plot(
                ts_data.index,
                ts_data["Avg_Engagement"],
                label="Historical",
                linewidth=2,
            )
            plt.plot(
                ts_data.index, ts_data["Engagement_Trend"], label="Trend", alpha=0.7
            )
            plt.plot(
                future_dates,
                forecast,
                label="Forecast",
                linestyle="--",
                linewidth=2,
                color="red",
            )
            plt.fill_between(
                future_dates, forecast * 0.8, forecast * 1.2, alpha=0.2, color="red"
            )
            plt.title("Engagement Time Series Analysis & Forecast")
            plt.xlabel("Date")
            plt.ylabel("Average Engagement")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def perform_causal_inference(self):
        """Perform causal inference to understand what truly drives engagement"""
        print("Performing causal inference analysis...")

        # Define potential treatment variables
        treatment_vars = [
            "Has Call to Action",
            "Is_Viral",
            "Is Weekend",
            "Is Peak Hour",
        ]
        available_treatments = [
            var for var in treatment_vars if var in self.data.columns
        ]

        if not available_treatments:
            print("No treatment variables available for causal analysis")
            return

        causal_results = {}

        for treatment in available_treatments:
            # Simple causal inference using matching/stratification
            treated = self.data[self.data[treatment] == 1]["Total Engagement"]
            control = self.data[self.data[treatment] == 0]["Total Engagement"]

            if len(treated) > 5 and len(control) > 5:
                # Calculate Average Treatment Effect (ATE)
                ate = treated.mean() - control.mean()

                # Statistical significance test
                t_stat, p_value = stats.ttest_ind(treated, control)

                causal_results[treatment] = {
                    "ATE": ate,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "treated_mean": treated.mean(),
                    "control_mean": control.mean(),
                    "treated_count": len(treated),
                    "control_count": len(control),
                }

        self.causal_insights = causal_results

        # Visualize causal effects
        if causal_results:
            effects = [results["ATE"] for results in causal_results.values()]
            treatments = list(causal_results.keys())
            significance = [
                "Significant" if results["significant"] else "Not Significant"
                for results in causal_results.values()
            ]

            plt.figure(figsize=(12, 6))
            colors = [
                "green" if sig == "Significant" else "orange" for sig in significance
            ]
            bars = plt.bar(treatments, effects, color=colors, alpha=0.7)
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.title("Causal Effects on Engagement (Average Treatment Effects)")
            plt.xlabel("Treatment Variables")
            plt.ylabel("Average Treatment Effect")
            plt.xticks(rotation=45)

            # Add significance indicators
            for bar, sig in zip(bars, significance):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (0.05 * abs(height)),
                    "***" if sig == "Significant" else "ns",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plt.show()

    def simulate_ab_tests(self):
        """Simulate A/B tests for content optimization"""
        print("Simulating A/B tests for content optimization...")

        # Define testable variables
        test_variables = {
            "Optimal_Hashtag_Count": ("Hashtag Count", "above_median"),
            "Long_vs_Short_Content": ("Description Length", "above_median"),
            "Peak_vs_Off_Hours": ("Is Peak Hour", "binary"),
            "Weekend_vs_Weekday": ("Is Weekend", "binary"),
        }

        ab_results = {}

        for test_name, (feature, test_type) in test_variables.items():
            if feature not in self.data.columns:
                continue

            if test_type == "above_median":
                median_val = self.data[feature].median()
                group_a = self.data[self.data[feature] > median_val]["Total Engagement"]
                group_b = self.data[self.data[feature] <= median_val][
                    "Total Engagement"
                ]
                group_a_label = f"High {feature}"
                group_b_label = f"Low {feature}"
            elif test_type == "binary":
                group_a = self.data[self.data[feature] == 1]["Total Engagement"]
                group_b = self.data[self.data[feature] == 0]["Total Engagement"]
                group_a_label = f"{feature} = Yes"
                group_b_label = f"{feature} = No"

            if len(group_a) > 5 and len(group_b) > 5:
                # Statistical test
                t_stat, p_value = stats.ttest_ind(group_a, group_b)
                effect_size = (group_a.mean() - group_b.mean()) / np.sqrt(
                    (
                        (len(group_a) - 1) * group_a.var()
                        + (len(group_b) - 1) * group_b.var()
                    )
                    / (len(group_a) + len(group_b) - 2)
                )

                # Confidence interval for difference
                pooled_se = np.sqrt(
                    (group_a.var() / len(group_a)) + (group_b.var() / len(group_b))
                )
                ci_lower = (group_a.mean() - group_b.mean()) - 1.96 * pooled_se
                ci_upper = (group_a.mean() - group_b.mean()) + 1.96 * pooled_se

                ab_results[test_name] = {
                    "group_a_mean": group_a.mean(),
                    "group_b_mean": group_b.mean(),
                    "difference": group_a.mean() - group_b.mean(),
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "significant": p_value < 0.05,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "group_a_label": group_a_label,
                    "group_b_label": group_b_label,
                    "recommendation": (
                        "Choose A" if group_a.mean() > group_b.mean() else "Choose B"
                    ),
                }

        self.ab_test_results = ab_results

        # Visualize A/B test results
        if ab_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            for i, (test_name, results) in enumerate(ab_results.items()):
                if i >= len(axes):
                    break

                ax = axes[i]
                groups = [results["group_a_label"], results["group_b_label"]]
                means = [results["group_a_mean"], results["group_b_mean"]]
                colors = [
                    "green" if results["significant"] else "blue" for _ in range(2)
                ]

                bars = ax.bar(groups, means, color=colors, alpha=0.7)
                ax.set_title(f'{test_name}\n(p={results["p_value"]:.3f})')
                ax.set_ylabel("Average Engagement")

                # Add significance indicator
                if results["significant"]:
                    ax.text(
                        0.5,
                        max(means) * 1.1,
                        "***",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=16,
                        color="red",
                    )

            plt.tight_layout()
            plt.show()

    def create_interactive_dashboard(self):
        """Create interactive dashboard with Plotly"""
        print("Creating interactive dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Engagement Over Time",
                "Content Performance by Type",
                "Optimal Posting Hours",
                "Feature Importance",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Time series plot
        if self.time_series_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.time_series_data.index,
                    y=self.time_series_data["Avg_Engagement"],
                    mode="lines+markers",
                    name="Daily Avg Engagement",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=self.time_series_data.index,
                    y=self.time_series_data["Posts_Count"],
                    mode="lines+markers",
                    name="Daily Post Count",
                    line=dict(color="red"),
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # Content type performance
        if "Post type" in self.data.columns:
            post_type_perf = (
                self.data.groupby("Post type")["Total Engagement"]
                .mean()
                .sort_values(ascending=True)
            )
            fig.add_trace(
                go.Bar(
                    x=post_type_perf.values,
                    y=post_type_perf.index,
                    name="Avg Engagement by Type",
                    orientation="h",
                    marker_color="lightblue",
                ),
                row=1,
                col=2,
            )

        # Hourly performance
        if "Hour" in self.data.columns:
            hourly_perf = self.data.groupby("Hour")["Total Engagement"].mean()
            fig.add_trace(
                go.Bar(
                    x=hourly_perf.index,
                    y=hourly_perf.values,
                    name="Engagement by Hour",
                    marker_color="lightgreen",
                ),
                row=2,
                col=1,
            )

        # Feature importance (if available)
        if self.feature_importance:
            # Get average importance across models
            all_features = set()
            for model_features in self.feature_importance.values():
                all_features.update(model_features.keys())

            avg_importance = {}
            for feature in all_features:
                importances = [
                    self.feature_importance[model].get(feature, 0)
                    for model in self.feature_importance.keys()
                ]
                avg_importance[feature] = np.mean(importances)

            sorted_features = sorted(
                avg_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            features, importance_values = zip(*sorted_features)

            fig.add_trace(
                go.Bar(
                    x=list(importance_values),
                    y=list(features),
                    name="Feature Importance",
                    orientation="h",
                    marker_color="orange",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Instagram Content Strategy Dashboard",
            showlegend=True,
        )

        # Show the interactive plot
        fig.show()

    def perform_correlation_network_analysis(self):
        """Analyze correlation networks between features"""
        print("Performing correlation network analysis...")

        # Select numerical features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        correlation_data = self.data[numeric_cols].fillna(0)

        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()

        # Create network-style correlation plot
        plt.figure(figsize=(16, 12))

        # Mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Network", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Identify strong correlations with engagement
        engagement_corrs = (
            corr_matrix["Total Engagement"].abs().sort_values(ascending=False)[1:11]
        )

        print("\nStrongest correlations with Total Engagement:")
        for feature, corr in engagement_corrs.items():
            print(f"  {feature}: {corr:.3f}")

        return corr_matrix

    def perform_clustering_analysis(self):
        """Perform clustering to identify content patterns"""
        print("Performing clustering analysis...")

        # Select numerical features for clustering
        cluster_features = [
            "Total Engagement",
            "Hashtag Count",
            "Word Count",
            "Sentiment",
            "Hour",
            "Is Weekend",
            "Description Length",
        ]

        # Filter available features
        available_features = [f for f in cluster_features if f in self.data.columns]

        if len(available_features) >= 3:
            cluster_data = self.data[available_features].fillna(0)

            # Standardize features
            cluster_data_scaled = self.scaler.fit_transform(cluster_data)

            # Determine optimal number of clusters using elbow method
            inertias = []
            k_range = range(2, min(8, len(self.data) // 2))

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(cluster_data_scaled)
                inertias.append(kmeans.inertia_)

            # Use elbow method or default to 4 clusters
            optimal_k = 4 if len(k_range) >= 3 else len(k_range)[0]

            # Perform clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            self.data["Content Cluster"] = kmeans.fit_predict(cluster_data_scaled)

            # Analyze clusters
            cluster_analysis = self.data.groupby("Content Cluster")[
                available_features
            ].mean()
            print("\nCluster Analysis:")
            print(cluster_analysis.round(2))

            # Visualize clusters (PCA for dimensionality reduction)
            if len(available_features) > 2:
                pca = PCA(n_components=2)
                cluster_pca = pca.fit_transform(cluster_data_scaled)

                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(
                    cluster_pca[:, 0],
                    cluster_pca[:, 1],
                    c=self.data["Content Cluster"],
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(scatter)
                plt.title("Content Clusters (PCA Visualization)")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
                plt.show()

    def prepare_features_for_modeling(self):
        """Prepare features for machine learning models"""
        print("Preparing features for modeling...")

        # Define target variable
        target = "Total Engagement"

        # Select features for modeling
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and ID-like columns
        exclude_features = [target, "Publish time"] + [
            col for col in numeric_features if "id" in col.lower()
        ]
        feature_columns = [
            col for col in numeric_features if col not in exclude_features
        ]

        # Handle categorical variables
        categorical_features = ["Post type", "Day of Week"]
        available_categorical = [
            f for f in categorical_features if f in self.data.columns
        ]

        # Create feature matrix
        X_numeric = self.data[feature_columns].fillna(0)

        # Add encoded categorical features
        if available_categorical:
            X_categorical = pd.get_dummies(
                self.data[available_categorical], prefix=available_categorical
            )
            X = pd.concat([X_numeric, X_categorical], axis=1)
        else:
            X = X_numeric

        y = self.data[target]

        return (
            X,
            y,
            feature_columns
            + (list(X_categorical.columns) if available_categorical else []),
        )

    def train_multiple_models(self, X, y, feature_names):
        """Train multiple ML models with hyperparameter tuning"""
        print("Training multiple models with hyperparameter tuning...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.scaler_columns = X_train.columns

        # Define models with hyperparameter grids
        model_configs = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
            },
            "XGBoost": {
                "model": xgb.XGBRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                },
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
            },
            "Ridge": {
                "model": Ridge(),
                "params": {"alpha": [0.1, 1.0, 10.0, 100.0]},
                "scaled": True,
            },
            "Neural Network": {
                "model": MLPRegressor(random_state=42, max_iter=1000),
                "params": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "alpha": [0.001, 0.01, 0.1],
                    "learning_rate_init": [0.001, 0.01],
                },
                "scaled": True,
            },
            "Gaussian Process": {
                "model": GaussianProcessRegressor(
                    kernel=ConstantKernel() * RBF(), random_state=42
                ),
                "params": {
                    "alpha": [1e-10, 1e-8, 1e-6],
                    "kernel": [
                        ConstantKernel() * RBF(),
                        ConstantKernel() * RBF() + ConstantKernel(),
                    ],
                },
                "scaled": True,
            },
            "SVR": {
                "model": SVR(),
                "params": {
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto"],
                    "kernel": ["rbf", "linear"],
                },
                "scaled": True,
            },
        }

        # Train and evaluate models
        results = {}

        for name, config in model_configs.items():
            print(f"\nTraining {name}...")

            # Choose appropriate data (scaled or not)
            X_train_current = X_train_scaled if config.get("scaled", False) else X_train
            X_test_current = X_test_scaled if config.get("scaled", False) else X_test

            # Grid search with cross-validation
            cv_folds = min(5, len(X_train) // 10) if len(X_train) > 50 else 3

            grid_search = GridSearchCV(
                config["model"],
                config["params"],
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )

            grid_search.fit(X_train_current, y_train)

            # Best model predictions
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_current)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                "model": best_model,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "best_params": grid_search.best_params_,
                "cv_score": -grid_search.best_score_,
            }

            # Store feature importance for tree-based models
            if hasattr(best_model, "feature_importances_"):
                self.feature_importance[name] = dict(
                    zip(feature_names, best_model.feature_importances_)
                )

            print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.3f}")

        self.models = results
        return results, X_test, y_test

    def create_ensemble_model(self, X_train, y_train):
        """Create an ensemble model combining best performers"""
        print("Creating ensemble model...")

        # Select top 3 models based on R² score
        sorted_models = sorted(
            self.models.items(), key=lambda x: x[1]["r2"], reverse=True
        )
        top_models = sorted_models[:3]

        # Create voting regressor
        estimators = [(name, results["model"]) for name, results in top_models]

        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(X_train, y_train)

        return ensemble

    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        print("Analyzing feature importance...")

        if not self.feature_importance:
            print("No feature importance data available.")
            return

        # Combine feature importance from all models
        all_features = set()
        for model_features in self.feature_importance.values():
            all_features.update(model_features.keys())

        # Calculate average importance across models
        avg_importance = {}
        for feature in all_features:
            importances = [
                self.feature_importance[model].get(feature, 0)
                for model in self.feature_importance.keys()
            ]
            avg_importance[feature] = np.mean(importances)

        # Sort by importance
        sorted_features = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Plot top features
        top_n = min(15, len(sorted_features))
        features, importance_values = zip(*sorted_features[:top_n])

        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(importance_values), y=list(features), palette="viridis")
        plt.title("Average Feature Importance Across Models")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()

    def model_interpretability(self):
        """model interpretability and SHAP-like analysis"""
        print("Performing model interpretability...")

        if not self.models:
            print("No models available for interpretability analysis")
            return

        # Get the best model
        best_model_name = max(self.models.items(), key=lambda x: x[1]["r2"])[0]
        best_model = self.models[best_model_name]["model"]

        # Prepare data for interpretation - USE THE SAME PREPROCESSING AS TRAINING
        X, y, feature_names = self.prepare_features_for_modeling()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Apply the same scaling that was used during training
        # Check if the model requires scaling based on the model configs
        scaled_models = ["Ridge", "Neural Network", "Gaussian Process", "SVR"]
        needs_scaling = best_model_name in scaled_models

        if needs_scaling:
            # Use the same scaler that was fitted during training
            X_train_filtered = X_train[self.scaler_columns]
            X_test_filtered = X_test[self.scaler_columns]
            X_test_processed = self.scaler.transform(X_test_filtered)
            feature_names = list(self.scaler_columns)
        else:
            X_test_processed = X_test
            feature_names = list(X_test.columns)

        # Ensure feature consistency by using the exact same features
        try:
            # Test prediction to verify feature compatibility
            test_pred = best_model.predict(X_test_processed[:1])

            # Permutation importance with consistent preprocessing
            perm_importance = permutation_importance(
                best_model, X_test_processed, y_test, n_repeats=10, random_state=42
            )

            # Create permutation importance plot
            perm_importance_df = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": perm_importance.importances_mean,
                        "std": perm_importance.importances_std,
                    }
                )
                .sort_values("importance", ascending=True)
                .tail(15)
            )

            plt.figure(figsize=(10, 8))
            plt.barh(
                perm_importance_df["feature"],
                perm_importance_df["importance"],
                xerr=perm_importance_df["std"],
                alpha=0.7,
            )
            plt.xlabel("Permutation Importance")
            plt.title(f"Permutation Feature Importance - {best_model_name}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in permutation importance calculation: {e}")
            print("Skipping permutation importance analysis")

        # Partial dependence analysis (simplified) - use original features for interpretation
        top_features = self.get_top_features(n=5)

        if top_features:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, (feature, _) in enumerate(top_features[:6]):
                if i >= len(axes):
                    break

                if feature in X.columns:
                    try:
                        # Create feature range
                        feature_range = np.linspace(
                            X[feature].min(), X[feature].max(), 50
                        )

                        # Calculate partial dependence
                        X_partial = X_test.copy()
                        partial_predictions = []

                        for val in feature_range:
                            X_partial_temp = X_partial.copy()
                            X_partial_temp[feature] = val

                            # Apply same preprocessing as used for the model
                            if needs_scaling:
                                X_partial_filtered = X_partial_temp[self.scaler_columns]
                                X_partial_processed = self.scaler.transform(
                                    X_partial_filtered
                                )
                            else:
                                X_partial_processed = X_partial_temp

                            pred = best_model.predict(X_partial_processed).mean()
                            partial_predictions.append(pred)

                        axes[i].plot(feature_range, partial_predictions, linewidth=2)
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel("Predicted Engagement")
                        axes[i].set_title(f"Partial Dependence: {feature}")
                        axes[i].grid(True, alpha=0.3)

                    except Exception as e:
                        print(
                            f"Error calculating partial dependence for {feature}: {e}"
                        )
                        axes[i].text(
                            0.5,
                            0.5,
                            f"Error: {feature}",
                            ha="center",
                            va="center",
                            transform=axes[i].transAxes,
                        )
            plt.suptitle(
                f"Partial Dependence Analysis - {best_model_name}", fontsize=16
            )
            plt.tight_layout()
            plt.show()
        print("Model interpretability analysis completed.")

    def get_top_features(self, n=10):
        """Get top N most important features across all models"""
        if not self.feature_importance:
            return []

        # Aggregate feature importance across models
        all_features = set()
        for model_features in self.feature_importance.values():
            all_features.update(model_features.keys())

        avg_importance = {}
        for feature in all_features:
            importances = [
                self.feature_importance[model].get(feature, 0)
                for model in self.feature_importance.keys()
            ]
            avg_importance[feature] = np.mean(importances)

        return sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:n]

    def perform_automated_feature_selection(self, X, y):
        """Automated feature selection using multiple techniques"""
        print("Performing automated feature selection...")

        feature_selection_results = {}

        # 1. Univariate feature selection
        selector_univariate = SelectKBest(score_func=f_regression, k=20)
        X_selected_univariate = selector_univariate.fit_transform(X, y)
        selected_features_univariate = X.columns[
            selector_univariate.get_support()
        ].tolist()
        feature_selection_results["Univariate"] = selected_features_univariate

        # 2. Recursive Feature Elimination with Cross-Validation
        if len(X.columns) > 10:  # Only if we have enough features
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector_rfe = RFECV(
                estimator, step=1, cv=3, scoring="r2", min_features_to_select=5
            )
            selector_rfe.fit(X, y)
            selected_features_rfe = X.columns[selector_rfe.support_].tolist()
            feature_selection_results["RFE-CV"] = selected_features_rfe

        # 3. L1 regularization feature selection
        lasso_selector = Lasso(alpha=0.1, random_state=42)
        lasso_selector.fit(self.scaler.fit_transform(X), y)
        selected_features_lasso = X.columns[
            np.abs(lasso_selector.coef_) > 0.01
        ].tolist()
        feature_selection_results["Lasso"] = selected_features_lasso

        # Find consensus features
        all_selected = []
        for features in feature_selection_results.values():
            all_selected.extend(features)

        feature_counts = pd.Series(all_selected).value_counts()
        consensus_features = feature_counts[feature_counts >= 2].index.tolist()

        print(f"Feature selection results:")
        for method, features in feature_selection_results.items():
            print(f"  {method}: {len(features)} features")
        print(f"  Consensus features: {len(consensus_features)}")

        return consensus_features, feature_selection_results

    def generate_insights_and_recommendations(self):
        """Generate actionable insights and recommendations"""
        print("\n" + "=" * 60)
        print("INSTAGRAM CONTENT STRATEGY INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)

        # 1. Best performing content analysis
        if "Content Cluster" in self.data.columns:
            cluster_performance = self.data.groupby("Content Cluster")[
                "Total Engagement"
            ].agg(["mean", "count"])
            best_cluster = cluster_performance["mean"].idxmax()
            print(f"\n BEST PERFORMING CONTENT CLUSTER: {best_cluster}")
            print(
                f"   Average Engagement: {cluster_performance.loc[best_cluster, 'mean']:.0f}"
            )
            print(
                f"   Number of Posts: {cluster_performance.loc[best_cluster, 'count']}"
            )

        # 2. Optimal posting times
        if "Hour" in self.data.columns:
            hourly_engagement = self.data.groupby("Hour")["Total Engagement"].mean()
            best_hours = hourly_engagement.nlargest(3)
            print(f"\n OPTIMAL POSTING HOURS:")
            for hour, engagement in best_hours.items():
                print(f"   {hour}:00 - Average Engagement: {engagement:.0f}")

        # 3. Day of week analysis
        if "Day of Week" in self.data.columns:
            daily_engagement = self.data.groupby("Day of Week")[
                "Total Engagement"
            ].mean()
            best_day = daily_engagement.idxmax()
            print(f"\n BEST DAY TO POST: {best_day}")
            print(f"   Average Engagement: {daily_engagement[best_day]:.0f}")

        # 4. Content length recommendations
        if "Description Length" in self.data.columns:
            length_engagement = self.data.groupby(
                pd.cut(self.data["Description Length"], bins=5)
            )["Total Engagement"].mean()
            print(f"\n CONTENT LENGTH INSIGHTS:")
            for length_range, engagement in length_engagement.items():
                if not pd.isna(engagement):
                    print(f"   {length_range}: {engagement:.0f} avg engagement")

        # 5. Hashtag recommendations
        if "Hashtag Count" in self.data.columns:
            hashtag_performance = self.data.groupby("Hashtag Count")[
                "Total Engagement"
            ].mean()
            optimal_hashtags = hashtag_performance.idxmax()
            print(f"\n OPTIMAL HASHTAG COUNT: {optimal_hashtags}")
            print(f"   Average Engagement: {hashtag_performance[optimal_hashtags]:.0f}")

        # 6. Model performance summary
        if self.models:
            best_model = max(self.models.items(), key=lambda x: x[1]["r2"])
            print(f"\n BEST PREDICTION MODEL: {best_model[0]}")
            print(f"   R² Score: {best_model[1]['r2']:.3f}")
            print(f"   RMSE: {best_model[1]['rmse']:.2f}")

        print("\n" + "=" * 60)
        print("ACTIONABLE RECOMMENDATIONS:")
        print("=" * 60)

        recommendations = [
            " Focus on content types and formats that align with your best-performing cluster",
            " Schedule posts during identified optimal hours for maximum engagement",
            " Prioritize posting on high-engagement days",
            " Optimize content length based on engagement patterns",
            " Use the optimal number of hashtags consistently",
            " A/B test content variations within successful patterns",
            " Monitor performance weekly and adjust strategy based on trends",
            " Engage with audience during peak activity hours",
            " Maintain consistency in posting schedule and content quality",
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Instagram Content Strategy Analysis...")
        print("=" * 60)

        # Load and preprocess data
        self.load_and_preprocess_data()

        # Advanced feature engineering
        self.feature_engineering()

        # Clustering analysis
        self.perform_clustering_analysis()

        # Prepare features for modeling
        X, y, feature_names = self.prepare_features_for_modeling()

        # Train multiple models
        results, X_test, y_test = self.train_multiple_models(X, y, feature_names)

        # Analyze feature importance
        important_features = self.analyze_feature_importance()

        # Generate insights and recommendations
        self.generate_insights_and_recommendations()

        return self


# Additional utility functions for post-analysis


def export_actionable_insights(analyzer, filename="instagram_insights.txt"):
    """Export actionable insights to a text file"""
    with open(filename, "w") as f:
        f.write("INSTAGRAM CONTENT STRATEGY INSIGHTS\n")
        f.write("=" * 50 + "\n\n")

        # Best posting times
        if "Hour" in analyzer.data.columns:
            hourly_perf = analyzer.data.groupby("Hour")["Total Engagement"].mean()
            best_hours = hourly_perf.nlargest(3)
            f.write("OPTIMAL POSTING TIMES:\n")
            for hour, engagement in best_hours.items():
                f.write(f"  {hour}:00 - Avg Engagement: {engagement:.0f}\n")
            f.write("\n")

        # Content recommendations
        if analyzer.feature_importance:
            f.write("KEY SUCCESS FACTORS:\n")
            top_features = analyzer.get_top_features(n=5)
            for feature, importance in top_features:
                f.write(f"  - {feature}: {importance:.3f}\n")
            f.write("\n")

        # A/B test recommendations
        if analyzer.ab_test_results:
            f.write("A/B TEST RECOMMENDATIONS:\n")
            for test_name, results in analyzer.ab_test_results.items():
                if results["significant"]:
                    f.write(f"  - {test_name}: {results['recommendation']}\n")
                    f.write(
                        f"    Effect: {results['difference']:.0f} engagement difference\n"
                    )
            f.write("\n")

    print(f" Insights exported to {filename}")


def create_content_calendar_template(analyzer):
    """Create a content calendar template based on insights"""
    optimal_times = []
    if "Hour" in analyzer.data.columns:
        hourly_perf = analyzer.data.groupby("Hour")["Total Engagement"].mean()
        optimal_times = hourly_perf.nlargest(3).index.tolist()

    template = {
        "monday": {"times": optimal_times, "content_type": "motivational"},
        "tuesday": {"times": optimal_times, "content_type": "educational"},
        "wednesday": {"times": optimal_times, "content_type": "behind_scenes"},
        "thursday": {"times": optimal_times, "content_type": "user_generated"},
        "friday": {"times": optimal_times, "content_type": "entertainment"},
        "saturday": {"times": optimal_times, "content_type": "lifestyle"},
        "sunday": {"times": optimal_times, "content_type": "inspirational"},
    }

    with open("content_calendar_template.json", "w") as f:
        json.dump(template, f, indent=2)

    print(" Content calendar template created: content_calendar_template.json")


# Usage Example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = InstagramAnalyzer("Instagram-Data.csv")

    # Run complete analysis
    analyzer.run_complete_analysis()

    # Additional visualizations
    plt.style.use("seaborn-v0_8")

    # Model comparison visualization
    if analyzer.models:
        model_names = list(analyzer.models.keys())
        r2_scores = [analyzer.models[name]["r2"] for name in model_names]
        rmse_scores = [analyzer.models[name]["rmse"] for name in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # R² scores
        ax1.bar(model_names, r2_scores, color="skyblue", alpha=0.7)
        ax1.set_title("Model Performance - R² Score")
        ax1.set_ylabel("R² Score")
        ax1.tick_params(axis="x", rotation=45)

        # RMSE scores
        ax2.bar(model_names, rmse_scores, color="lightcoral", alpha=0.7)
        ax2.set_title("Model Performance - RMSE")
        ax2.set_ylabel("RMSE")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    # Additional advanced analyses that weren't called in the main pipeline
    print("\n" + "=" * 60)
    print("ANALYSES")
    print("=" * 60)

    # Anomaly detection analysis
    analyzer.detect_anomalies_and_outliers()

    # Time series analysis with forecasting
    analyzer.perform_time_series_analysis()

    # Causal inference analysis
    analyzer.perform_causal_inference()

    # A/B testing simulation
    analyzer.simulate_ab_tests()

    # Correlation network analysis
    correlation_matrix = analyzer.perform_correlation_network_analysis()

    # Advanced model interpretability
    analyzer.model_interpretability()

    # Interactive dashboard creation
    analyzer.create_interactive_dashboard()

    # Automated feature selection
    X, y, feature_names = analyzer.prepare_features_for_modeling()
    consensus_features, selection_results = (
        analyzer.perform_automated_feature_selection(X, y)
    )

    print(f"\n FEATURE SELECTION RESULTS:")
    print(f"   Total original features: {len(feature_names)}")
    print(f"   Consensus important features: {len(consensus_features)}")
    print(f"   Selected features: {consensus_features[:10]}")  # Show top 10

    # Create ensemble model with selected features
    if consensus_features:
        X_selected = X[consensus_features]
        X_train_selected, X_test_selected, y_train_selected, y_test_selected = (
            train_test_split(X_selected, y, test_size=0.2, random_state=42)
        )

        ensemble_model = analyzer.create_ensemble_model(
            X_train_selected, y_train_selected
        )
        ensemble_predictions = ensemble_model.predict(X_test_selected)

        ensemble_r2 = r2_score(y_test_selected, ensemble_predictions)
        ensemble_rmse = np.sqrt(
            mean_squared_error(y_test_selected, ensemble_predictions)
        )

        print(f"\n ENSEMBLE MODEL PERFORMANCE:")
        print(f"   R² Score: {ensemble_r2:.3f}")
        print(f"   RMSE: {ensemble_rmse:.2f}")

        # Compare ensemble with best individual model
        best_individual_r2 = max(analyzer.models.values(), key=lambda x: x["r2"])["r2"]
        improvement = ((ensemble_r2 - best_individual_r2) / best_individual_r2) * 100
        print(f"   Improvement over best individual model: {improvement:.1f}%")

    # Final summary and export recommendations
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - FINAL SUMMARY")
    print("=" * 60)

    total_posts = len(analyzer.data)
    avg_engagement = analyzer.data["Total Engagement"].mean()
    engagement_std = analyzer.data["Total Engagement"].std()
    viral_posts = (
        analyzer.data["Is_Viral"].sum() if "Is_Viral" in analyzer.data.columns else 0
    )

    print(f"   DATASET OVERVIEW:")
    print(f"   Total posts analyzed: {total_posts}")
    print(f"   Average engagement per post: {avg_engagement:.0f}")
    print(f"   Engagement standard deviation: {engagement_std:.0f}")
    print(f"   Viral posts identified: {viral_posts}")

    # Top insights summary
    print(f"\n KEY INSIGHTS:")

    # Best performing features
    if analyzer.feature_importance:
        top_features = analyzer.get_top_features(n=5)
        print(f"   Top 5 engagement drivers:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"     {i}. {feature} (importance: {importance:.3f})")

    # Performance patterns
    if "Hour" in analyzer.data.columns:
        peak_hour = analyzer.data.groupby("Hour")["Total Engagement"].mean().idxmax()
        print(f"   Peak engagement hour: {peak_hour}:00")

    if "Is Weekend" in analyzer.data.columns:
        weekend_performance = analyzer.data.groupby("Is Weekend")[
            "Total Engagement"
        ].mean()
        weekend_boost = (
            (weekend_performance[1] - weekend_performance[0]) / weekend_performance[0]
        ) * 100
        weekend_status = "better" if weekend_boost > 0 else "worse"
        print(f"   Weekend posts perform {abs(weekend_boost):.1f}% {weekend_status}")

    # Model reliability
    if analyzer.models:
        model_scores = [model["r2"] for model in analyzer.models.values()]
        avg_model_score = np.mean(model_scores)
        model_consistency = 1 - (np.std(model_scores) / np.mean(model_scores))
        print(
            f"   Model prediction reliability: {avg_model_score:.3f} (consistency: {model_consistency:.3f})"
        )

    print(f"\n NEXT STEPS:")
    next_steps = [
        "Implement posting schedule based on optimal timing insights",
        "Create content templates following high-performing patterns",
        "Set up automated monitoring of engagement metrics",
        "Plan A/B tests for content variations",
        "Establish monthly performance review cycles",
        "Consider seasonal adjustments to content strategy",
        "Monitor competitor activities and market trends",
    ]

    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")

    # Save results summary to file (optional)
    try:
        summary_data = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_posts": int(total_posts),
            "avg_engagement": float(avg_engagement),
            "viral_posts": int(viral_posts) if viral_posts else 0,
            "best_model": (
                max(analyzer.models.items(), key=lambda x: x[1]["r2"])[0]
                if analyzer.models
                else None
            ),
            "best_model_r2": (
                max(analyzer.models.values(), key=lambda x: x["r2"])["r2"]
                if analyzer.models
                else None
            ),
            "top_features": (
                [feature for feature, _ in analyzer.get_top_features(n=5)]
                if analyzer.feature_importance
                else []
            ),
            "causal_insights": analyzer.causal_insights,
            "ab_test_results": analyzer.ab_test_results,
        }

        with open("instagram_analysis_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"\n Analysis summary saved to 'instagram_analysis_summary.json'")

    except Exception as e:
        print(f"\n Could not save summary file: {e}")

    print(f"\n ANALYSIS COMPLETE!")
    print(f"   Total execution time: Analysis completed successfully")
    print(f"   Generated insights: Ready for implementation")
    print("\n Running comprehensive Instagram analysis...")

    # Export additional insights
    export_actionable_insights(analyzer)
    create_content_calendar_template(analyzer)
    print(" Check generated files for detailed insights and recommendations.")
