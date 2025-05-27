import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Instagram Content Strategy Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DA1F2;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B4B4B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F7F9FA;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-text {
        background-color: #E1F5FE;
        border-left: 5px solid #039BE5;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and intro
st.markdown("<h1 class='main-header'>📊 Instagram Content Strategy Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload your Instagram data to get actionable insights and optimize your content strategy</p>", unsafe_allow_html=True)

class InstagramAnalyzer:
    def __init__(self, data):
        """Initialize the analyzer with the provided DataFrame"""
        self.data = data
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.time_series_data = None
        
    def preprocess_data(self):
        """Perform initial preprocessing"""
        # Convert 'Publish time' to datetime
        self.data["Publish time"] = pd.to_datetime(self.data["Publish time"])
        
        # Basic datetime features
        self.data["Day of Week"] = self.data["Publish time"].dt.day_name()
        self.data["Hour"] = self.data["Publish time"].dt.hour
        self.data["Month"] = self.data["Publish time"].dt.month
        self.data["Day of Month"] = self.data["Publish time"].dt.day
        self.data["Quarter"] = self.data["Publish time"].dt.quarter
        self.data["Is Weekend"] = (self.data["Publish time"].dt.weekday.isin([5, 6])).astype(int)
        
        # Calculate engagement metrics
        engagement_metrics = [col for col in ["Likes", "Shares", "Comments", "Saves", "Follows"] 
                             if col in self.data.columns]
        
        if engagement_metrics:
            self.data["Total Engagement"] = self.data[engagement_metrics].sum(axis=1)
            
            # Engagement rate (relative to followers if available)
            if "Followers" in self.data.columns:
                self.data["Engagement Rate"] = self.data["Total Engagement"] / self.data["Followers"]
        
        return self.data
    
    def feature_engineering(self):
        """Perform feature engineering"""
        # Text Analysis Features
        if "Description" in self.data.columns:
            # Basic text metrics
            self.data["Description Length"] = self.data["Description"].str.len()
            self.data["Word Count"] = self.data["Description"].str.split().str.len()
            self.data["Hashtag Count"] = self.data["Description"].str.count("#")
            self.data["Mention Count"] = self.data["Description"].str.count("@")
            
            # Sentiment analysis
            self.data["Sentiment"] = self.data["Description"].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
            )
            self.data["Subjectivity"] = self.data["Description"].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0
            )
            
            # Content type indicators
            self.data["Has Question"] = self.data["Description"].str.contains(
                r"\?", case=False, na=False
            ).astype(int)
            self.data["Has Call to Action"] = self.data["Description"].str.contains(
                r"\b(follow|like|share|comment|save|swipe|click|link|bio)\b", 
                case=False, na=False
            ).astype(int)
        
        # Temporal Features
        self.data["Hour Sin"] = np.sin(2 * np.pi * self.data["Hour"] / 24)
        self.data["Hour Cos"] = np.cos(2 * np.pi * self.data["Hour"] / 24)
        self.data["Is Peak Hour"] = self.data["Hour"].isin([8, 9, 12, 17, 18, 19, 20, 21]).astype(int)
        
        # Historical performance features
        self.data = self.data.sort_values("Publish time")
        self.data["Prev Post Engagement"] = self.data["Total Engagement"].shift(1).fillna(0)
        self.data["Avg Last 3 Posts"] = self.data["Total Engagement"].rolling(window=3, min_periods=1).mean().shift(1).fillna(0)
        
        return self.data
    
    def detect_anomalies(self):
        """Detect anomalous posts"""
        # Select numerical features for anomaly detection
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data = self.data[numeric_features].fillna(0)
        
        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = self.anomaly_detector.fit_predict(feature_data)
        
        self.data["Is_Anomaly"] = (anomaly_labels == -1).astype(int)
        
        # Viral content detection (top 5% engagement)
        engagement_threshold = self.data["Total Engagement"].quantile(0.95)
        self.data["Is_Viral"] = (self.data["Total Engagement"] >= engagement_threshold).astype(int)
        
        return self.data
    
    def perform_clustering(self):
        """Perform clustering to identify content patterns"""
        # Select numerical features for clustering
        cluster_features = [col for col in [
            "Total Engagement", "Hashtag Count", "Word Count", 
            "Sentiment", "Hour", "Is Weekend", "Description Length"
        ] if col in self.data.columns]
        
        # Need at least 3 features for meaningful clustering
        if len(cluster_features) >= 3:
            cluster_data = self.data[cluster_features].fillna(0)
            
            # Standardize features
            cluster_data_scaled = self.scaler.fit_transform(cluster_data)
            
            # Determine optimal number of clusters (or use fixed number)
            n_clusters = min(4, len(self.data) // 5)  # At least 5 points per cluster
            n_clusters = max(2, n_clusters)  # At least 2 clusters
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.data["Content Cluster"] = kmeans.fit_predict(cluster_data_scaled)
            
            return self.data, cluster_features
        
        return self.data, []
    
    def time_series_analysis(self):
        """Perform time series analysis"""
        # Create time series data
        ts_data = self.data.set_index("Publish time").resample("D")["Total Engagement"].agg(["mean", "sum", "count"])
        ts_data.columns = ["Avg_Engagement", "Total_Daily_Engagement", "Posts_Count"]
        ts_data = ts_data.fillna(0)
        
        self.time_series_data = ts_data
        
        # Trend analysis
        ts_data["Engagement_Trend"] = ts_data["Avg_Engagement"].rolling(window=7).mean()
        
        return ts_data
    
    def train_model(self):
        """Train a simple predictive model for engagement"""
        # Define target variable
        target = "Total Engagement"
        
        # Select features for modeling
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_features = [target, "Is_Anomaly", "Is_Viral", "Content Cluster"] 
        feature_columns = [col for col in numeric_features if col not in exclude_features 
                          and not (isinstance(col, str) and "time" in col.lower())]
        
        # Create feature matrix
        X = self.data[feature_columns].fillna(0)
        y = self.data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store feature importance
        importance = model.feature_importances_
        self.feature_importance = dict(zip(feature_columns, importance))
        
        results = {
            "model": model,
            "r2": r2,
            "rmse": rmse,
            "feature_names": feature_columns
        }
        
        self.models["Random Forest"] = results
        return results
    
    def generate_insights(self):
        """Generate key insights and recommendations"""
        insights = {}
        
        # Best performing content
        if "Content Cluster" in self.data.columns:
            cluster_performance = self.data.groupby("Content Cluster")["Total Engagement"].mean()
            best_cluster = cluster_performance.idxmax()
            insights["best_cluster"] = {
                "cluster": best_cluster,
                "engagement": cluster_performance[best_cluster]
            }
        
        # Optimal posting times
        if "Hour" in self.data.columns:
            hourly_engagement = self.data.groupby("Hour")["Total Engagement"].mean()
            best_hours = hourly_engagement.nlargest(3).index.tolist()
            insights["best_hours"] = {
                "hours": best_hours,
                "engagements": [hourly_engagement[hour] for hour in best_hours]
            }
        
        # Day of week analysis
        if "Day of Week" in self.data.columns:
            daily_engagement = self.data.groupby("Day of Week")["Total Engagement"].mean()
            best_day = daily_engagement.idxmax()
            insights["best_day"] = {
                "day": best_day,
                "engagement": daily_engagement[best_day]
            }
        
        # Top features
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            insights["top_features"] = top_features
        
        # Hashtag analysis
        if "Hashtag Count" in self.data.columns:
            hashtag_performance = self.data.groupby("Hashtag Count")["Total Engagement"].mean()
            optimal_hashtags = hashtag_performance.idxmax()
            insights["optimal_hashtags"] = {
                "count": optimal_hashtags,
                "engagement": hashtag_performance[optimal_hashtags]
            }
        
        # Viral post characteristics
        if "Is_Viral" in self.data.columns:
            viral_posts = self.data[self.data["Is_Viral"] == 1]
            if len(viral_posts) > 0:
                viral_insights = {}
                
                # Time patterns
                if "Hour" in viral_posts.columns:
                    viral_hours = viral_posts["Hour"].value_counts().head(3).index.tolist()
                    viral_insights["hours"] = viral_hours
                
                # Content length
                if "Description Length" in viral_posts.columns:
                    avg_viral_length = viral_posts["Description Length"].mean()
                    viral_insights["avg_length"] = avg_viral_length
                
                # Hashtag usage
                if "Hashtag Count" in viral_posts.columns:
                    avg_hashtags = viral_posts["Hashtag Count"].mean()
                    viral_insights["avg_hashtags"] = avg_hashtags
                
                insights["viral_characteristics"] = viral_insights
        
        return insights

def create_download_link(df, filename="data.csv"):
    """Create a download link for the DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download processed data</a>'
    return href

def download_fig(fig, filename="plot.png"):
    """Create a download link for matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download this plot</a>'
    return href

# Sidebar
st.sidebar.header("🔧 Analysis Options")

# Sample data option
use_sample_data = st.sidebar.checkbox("Use sample data", value=False, 
                                      help="Use sample Instagram data instead of uploading your own")

# File upload
uploaded_file = None
if not use_sample_data:
    uploaded_file = st.sidebar.file_uploader("Upload Instagram data CSV", type=["csv"])
    
    if not uploaded_file:
        st.info("👆 Upload your Instagram data CSV file or check 'Use sample data'")
        
        # Show expected data format
        st.markdown("### Expected data format:")
        st.markdown("""
        Your CSV should include these columns:
        - `Publish time`: When the post was published
        - `Likes`, `Comments`, `Shares`, etc.: Engagement metrics
        - `Description`: Post text content (optional)
        """)
        
        sample_data = {
            "Publish time": ["2023-01-01 09:30:00", "2023-01-02 15:45:00"],
            "Description": ["Check out our new product! #awesome", "Behind the scenes look at our workshop"],
            "Likes": [150, 245],
            "Comments": [12, 30],
            "Shares": [5, 15]
        }
        st.dataframe(pd.DataFrame(sample_data))
        
        # Early exit if no file and no sample data
        if not use_sample_data:
            st.stop()

# Analysis components to run
st.sidebar.subheader("Choose Analysis Components")
run_clustering = st.sidebar.checkbox("Content Clustering", value=True)
run_time_analysis = st.sidebar.checkbox("Time Series Analysis", value=True)
run_predictive_model = st.sidebar.checkbox("Predictive Modeling", value=True)
run_viral_analysis = st.sidebar.checkbox("Viral Content Analysis", value=True)

# Load data
@st.cache_data
def load_sample_data():
    """Load sample Instagram data"""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    likes = np.random.normal(loc=500, scale=200, size=100).astype(int)
    likes = np.abs(likes)  # Make all positive
    
    comments = np.random.normal(loc=50, scale=25, size=100).astype(int)
    comments = np.abs(comments)
    
    shares = np.random.normal(loc=20, scale=15, size=100).astype(int)
    shares = np.abs(shares)
    
    descriptions = [
        "Check out our new product launch! #excited #new",
        "Behind the scenes of our photoshoot today #bts #photoshoot",
        "Happy Monday! Start your week right with our tips",
        "Customer spotlight: See how @customer uses our product",
        "Tutorial: How to get the most out of our service"
    ] * 20
    
    hours = np.random.choice(range(7, 22), size=100)
    
    # Generate time with selected hours
    times = [date.replace(hour=hour) for date, hour in zip(dates, hours)]
    
    # Create sample data with some patterns
    # More engagement on weekends
    weekend_boost = [1.5 if pd.Timestamp(t).dayofweek >= 5 else 1.0 for t in times]
    
    # More engagement at certain hours (9am, 12pm, 6pm)
    hour_boost = [1.3 if pd.Timestamp(t).hour in [9, 12, 18] else 1.0 for t in times]
    
    # Combine boosts
    total_boost = [w * h for w, h in zip(weekend_boost, hour_boost)]
    
    # Apply boosts
    likes = [int(l * b) for l, b in zip(likes, total_boost)]
    
    # Create DataFrame
    data = pd.DataFrame({
        "Publish time": times,
        "Description": descriptions[:100],
        "Likes": likes,
        "Comments": comments,
        "Shares": shares,
    })
    
    # Add some hashtag patterns
    data["Description"] = data["Description"].astype(str)
    
    return data

if use_sample_data:
    data = load_sample_data()
    st.success("Using sample data")
else:
    data = pd.read_csv(uploaded_file)
    st.success(f"Data loaded successfully: {data.shape[0]} rows and {data.shape[1]} columns")

# Initialize analyzer
analyzer = InstagramAnalyzer(data)

# Data preprocessing and basic analysis
with st.spinner("Preprocessing data..."):
    analyzer.preprocess_data()
    analyzer.feature_engineering()

# Show processed data
with st.expander("View processed data"):
    st.dataframe(analyzer.data)
    st.markdown(create_download_link(analyzer.data, "instagram_processed_data.csv"), unsafe_allow_html=True)

# Basic statistics
st.markdown("<h2 class='sub-header'>📊 Engagement Overview</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Posts", f"{analyzer.data.shape[0]:,}")

with col2:
    if "Total Engagement" in analyzer.data.columns:
        avg_engagement = analyzer.data["Total Engagement"].mean()
        st.metric("Avg. Engagement", f"{avg_engagement:.1f}")

with col3:
    if "Likes" in analyzer.data.columns:
        avg_likes = analyzer.data["Likes"].mean()
        st.metric("Avg. Likes", f"{avg_likes:.1f}")

with col4:
    if "Comments" in analyzer.data.columns:
        avg_comments = analyzer.data["Comments"].mean()
        st.metric("Avg. Comments", f"{avg_comments:.1f}")

# Engagement trends over time
st.markdown("<h2 class='sub-header'>📈 Engagement Trends</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Engagement by day of week
    if "Day of Week" in analyzer.data.columns and "Total Engagement" in analyzer.data.columns:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = analyzer.data.groupby("Day of Week")["Total Engagement"].mean().reindex(day_order)
        
        fig = px.bar(
            x=day_data.index, 
            y=day_data.values,
            labels={'x': 'Day of Week', 'y': 'Average Engagement'},
            title="Engagement by Day of Week",
            color=day_data.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Engagement by hour
    if "Hour" in analyzer.data.columns and "Total Engagement" in analyzer.data.columns:
        hour_data = analyzer.data.groupby("Hour")["Total Engagement"].mean()
        
        fig = px.line(
            x=hour_data.index, 
            y=hour_data.values,
            labels={'x': 'Hour of Day', 'y': 'Average Engagement'},
            title="Engagement by Hour of Day",
            markers=True
        )
        
        # Add peak hours highlight
        peak_hours = [8, 9, 12, 17, 18, 19, 20]
        for hour in peak_hours:
            if hour in hour_data.index:
                fig.add_vline(x=hour, line_dash="dash", line_color="rgba(255, 0, 0, 0.3)")
        
        st.plotly_chart(fig, use_container_width=True)

# Run advanced analyses based on sidebar selections
st.markdown("<h2 class='sub-header'>🔍 Advanced Analysis</h2>", unsafe_allow_html=True)

# 1. Clustering Analysis
if run_clustering:
    st.subheader("Content Clustering")
    
    with st.spinner("Performing content clustering..."):
        cluster_data, cluster_features = analyzer.perform_clustering()
        
        if "Content Cluster" in cluster_data.columns:
            # Show clusters
            cluster_counts = cluster_data["Content Cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            
            # Cluster metrics
            cluster_metrics = cluster_data.groupby("Content Cluster")["Total Engagement"].mean().reset_index()
            cluster_metrics.columns = ["Cluster", "Avg Engagement"]
            
            # Best cluster
            best_cluster = cluster_metrics["Avg Engagement"].idxmax()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    cluster_counts, 
                    values="Count", 
                    names="Cluster",
                    title="Post Distribution by Cluster",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    cluster_metrics,
                    x="Cluster",
                    y="Avg Engagement",
                    title="Engagement by Content Cluster",
                    color="Avg Engagement",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("Cluster Characteristics")
            cluster_profiles = cluster_data.groupby("Content Cluster")[cluster_features].mean().round(2)
            st.dataframe(cluster_profiles)
            
            # Insights
            st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
            st.markdown(f"✨ **Key Insight**: Content in Cluster {cluster_metrics['Cluster'][best_cluster]} performs best with average engagement of {cluster_metrics['Avg Engagement'][best_cluster]:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)

# 2. Time Series Analysis
if run_time_analysis:
    st.subheader("Time Series Analysis")
    
    with st.spinner("Performing time series analysis..."):
        ts_data = analyzer.time_series_analysis()
        
        # Plot time series
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=ts_data.index, 
                y=ts_data["Avg_Engagement"],
                mode="lines+markers",
                name="Average Engagement",
                line=dict(color="blue")
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=ts_data.index, 
                y=ts_data["Posts_Count"],
                mode="lines+markers",
                name="Post Count",
                line=dict(color="red")
            ),
            secondary_y=True
        )
        
        if "Engagement_Trend" in ts_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=ts_data.index, 
                    y=ts_data["Engagement_Trend"],
                    mode="lines",
                    name="7-Day Trend",
                    line=dict(color="green", dash="dash")
                ),
                secondary_y=False
            )
        
        fig.update_layout(
            title="Engagement Over Time",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Engagement", secondary_y=False)
        fig.update_yaxes(title_text="Post Count", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly patterns
        weekly_data = analyzer.data.copy()
        weekly_data["Day Name"] = weekly_data["Publish time"].dt.day_name()
        weekly_data["Week Hour"] = weekly_data["Publish time"].dt.dayofweek * 24 + weekly_data["Hour"]
        
        weekly_engagement = weekly_data.groupby("Week Hour")["Total Engagement"].mean().reset_index()
        
        week_hours_labels = []
        for wh in weekly_engagement["Week Hour"]:
            day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][wh // 24]
            hour = wh % 24
            week_hours_labels.append(f"{day} {hour}:00")
        
        weekly_engagement["Label"] = week_hours_labels
        
        # Heatmap for weekly engagement patterns
        week_matrix = np.zeros((7, 24))
        for i, row in weekly_engagement.iterrows():
            day_idx = row["Week Hour"] // 24
            hour_idx = row["Week Hour"] % 24
            if day_idx < 7 and hour_idx < 24:  # Ensure within bounds
                week_matrix[day_idx, hour_idx] = row["Total Engagement"]
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hours = list(range(24))
        
        fig = px.imshow(
            week_matrix, 
            labels=dict(x="Hour of Day", y="Day of Week", color="Engagement"),
            x=hours,
            y=days,
            color_continuous_scale="Viridis",
            title="Weekly Engagement Heatmap"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        best_day_hour = weekly_engagement.loc[weekly_engagement["Total Engagement"].idxmax()]
        
        st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
        st.markdown(f"📅 **Optimal Posting Time**: {best_day_hour['Label']} shows the highest average engagement of {best_day_hour['Total Engagement']:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

# 3. Predictive Modeling
if run_predictive_model:
    st.subheader("Engagement Prediction Model")
    
    with st.spinner("Training predictive model..."):
        # Detect anomalies first
        analyzer.detect_anomalies()
        
        # Train model
        model_results = analyzer.train_model()
        
        if model_results and model_results["feature_names"]:
            # Feature importance
            importance_df = pd.DataFrame({
                "Feature": model_results["feature_names"],
                "Importance": model_results["model"].feature_importances_
            }).sort_values("Importance", ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model performance
                st.metric("Model R² Score", f"{model_results['r2']:.3f}")
                st.metric("Model RMSE", f"{model_results['rmse']:.1f}")
                
                # Top features text
                st.markdown("#### Top Factors for Engagement:")
                for i, (feature, importance) in enumerate(zip(importance_df["Feature"].head(5), 
                                                             importance_df["Importance"].head(5))):
                    st.markdown(f"{i+1}. **{feature}**: {importance:.3f}")
            
            with col2:
                # Feature importance chart
                top_features = importance_df.head(10)
                fig = px.bar(
                    top_features, 
                    x="Importance", 
                    y="Feature",
                    orientation="h",
                    title="Top 10 Engagement Drivers",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Engagement prediction tooltip
            st.info("💡 These factors are the best predictors of high engagement. Focus your content strategy on optimizing these elements.")

# 4. Viral Content Analysis
if run_viral_analysis:
    st.subheader("Viral Content Analysis")
    
    with st.spinner("Analyzing top-performing content..."):
        # Make sure anomaly detection has run
        if "Is_Viral" not in analyzer.data.columns:
            analyzer.detect_anomalies()
        
        viral_posts = analyzer.data[analyzer.data["Is_Viral"] == 1].copy()
        
        if len(viral_posts) > 0:
            st.write(f"Found {len(viral_posts)} viral posts (top 5% by engagement)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Viral post timing
                if "Hour" in viral_posts.columns:
                    viral_hours = viral_posts["Hour"].value_counts().reset_index()
                    viral_hours.columns = ["Hour", "Count"]
                    
                    fig = px.bar(
                        viral_hours,
                        x="Hour",
                        y="Count",
                        title="Viral Posts by Hour",
                        color="Count",
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Viral post day of week
                if "Day of Week" in viral_posts.columns:
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    viral_days = viral_posts["Day of Week"].value_counts().reindex(day_order).reset_index()
                    viral_days.columns = ["Day", "Count"]
                    viral_days = viral_days.fillna(0)
                    
                    fig = px.bar(
                        viral_days,
                        x="Day",
                        y="Count",
                        title="Viral Posts by Day",
                        color="Count",
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Characteristics of viral posts
            st.subheader("Viral Post Characteristics")
            
            # Select metrics to compare
            compare_cols = [col for col in [
                "Description Length", "Word Count", "Hashtag Count", 
                "Sentiment", "Has Call to Action", "Is Weekend", "Is Peak Hour"
            ] if col in analyzer.data.columns]
            
            if compare_cols:
                # Compare viral vs non-viral
                viral_metrics = viral_posts[compare_cols].mean()
                non_viral_metrics = analyzer.data[analyzer.data["Is_Viral"] == 0][compare_cols].mean()
                
                compare_df = pd.DataFrame({
                    "Metric": compare_cols,
                    "Viral": viral_metrics.values,
                    "Non-Viral": non_viral_metrics.values
                })
                
                # Calculate percent difference
                compare_df["% Difference"] = ((compare_df["Viral"] - compare_df["Non-Viral"]) / 
                                             compare_df["Non-Viral"] * 100).round(1)
                
                # Plot comparison
                fig = go.Figure()
                
                for metric in compare_cols:
                    viral_val = compare_df.loc[compare_df["Metric"] == metric, "Viral"].values[0]
                    non_viral_val = compare_df.loc[compare_df["Metric"] == metric, "Non-Viral"].values[0]
                    
                    # Normalize for better comparison
                    max_val = max(viral_val, non_viral_val)
                    viral_norm = viral_val / max_val if max_val > 0 else 0
                    non_viral_norm = non_viral_val / max_val if max_val > 0 else 0
                    
                    fig.add_trace(go.Bar(
                        x=[metric], 
                        y=[viral_norm],
                        name="Viral",
                        marker_color="crimson"
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=[metric], 
                        y=[non_viral_norm],
                        name="Non-Viral",
                        marker_color="lightskyblue"
                    ))
                
                fig.update_layout(
                    barmode="group",
                    title="Viral vs Non-Viral Content Comparison (Normalized)",
                    xaxis_title="Metrics",
                    yaxis_title="Normalized Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw metrics
                st.dataframe(compare_df)
                
                # Viral content insights
                significant_diffs = compare_df[abs(compare_df["% Difference"]) > 20]
                
                if not significant_diffs.empty:
                    st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
                    st.markdown("🔥 **Viral Content Insights:**")
                    
                    for _, row in significant_diffs.iterrows():
                        direction = "more" if row["% Difference"] > 0 else "less"
                        st.markdown(f"- Viral posts have {abs(row['% Difference']):.1f}% {direction} **{row['Metric']}** than average posts")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No viral posts detected in the dataset.")

# Generate overall insights
st.markdown("<h2 class='sub-header'>🌟 Content Strategy Recommendations</h2>", unsafe_allow_html=True)

with st.spinner("Generating insights and recommendations..."):
    insights = analyzer.generate_insights()
    
    if insights:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### When to Post")
            
            if "best_hours" in insights:
                hours = insights["best_hours"]["hours"]
                st.markdown(f"🕒 **Best Hours**: {', '.join([f'{h}:00' for h in hours])}")
            
            if "best_day" in insights:
                st.markdown(f"📅 **Best Day**: {insights['best_day']['day']}")
            
        with col2:
            st.markdown("### Content Optimization")
            
            if "optimal_hashtags" in insights:
                st.markdown(f"🔖 **Optimal Hashtag Count**: {insights['optimal_hashtags']['count']}")
            
            if "top_features" in insights:
                top_feature = insights["top_features"][0][0]
                st.markdown(f"🔑 **Key Engagement Driver**: {top_feature}")
        
        # Final recommendations
        st.markdown("### 🚀 Actionable Recommendations")
        
        recommendations = [
            "Schedule your posts during optimal hours identified for highest engagement",
            "Focus on content quality and user engagement rather than posting frequency",
            "Use the optimal number of hashtags for better discoverability",
            "Create more content similar to your top-performing cluster",
            "Analyze your viral posts to identify patterns and replicate success",
            "Maintain consistency in your posting schedule",
            "Engage with your audience by responding to comments quickly",
            "Monitor engagement metrics weekly to refine your strategy"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

# Export options
st.markdown("<h2 class='sub-header'>📤 Export Results</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Processed Data"):
        st.markdown(create_download_link(analyzer.data, "instagram_analyzed_data.csv"), unsafe_allow_html=True)

with col2:
    if st.button("Export Content Calendar Template"):
        # Create content calendar template
        if "Hour" in analyzer.data.columns:
            hourly_perf = analyzer.data.groupby("Hour")["Total Engagement"].mean()
            optimal_hours = hourly_perf.nlargest(3).index.tolist()
        else:
            optimal_hours = [9, 12, 18]  # Default values
            
        template = {
            "monday": {"times": optimal_hours, "content_type": "motivational"},
            "tuesday": {"times": optimal_hours, "content_type": "educational"},
            "wednesday": {"times": optimal_hours, "content_type": "behind_scenes"},
            "thursday": {"times": optimal_hours, "content_type": "user_generated"},
            "friday": {"times": optimal_hours, "content_type": "entertainment"},
            "saturday": {"times": optimal_hours, "content_type": "lifestyle"},
            "sunday": {"times": optimal_hours, "content_type": "inspirational"},
        }
        
        template_df = pd.DataFrame()
        for day, details in template.items():
            for time in details["times"]:
                template_df = pd.concat([template_df, pd.DataFrame({
                    "Day": [day.capitalize()],
                    "Hour": [f"{time}:00"],
                    "Content Type": [details["content_type"].replace("_", " ").title()],
                    "Notes": [""]
                })])
                
        st.markdown(create_download_link(template_df.reset_index(drop=True), 
                                        "instagram_content_calendar.csv"), 
                    unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Instagram Content Strategy Analyzer || Data analysis based on engagement patterns")