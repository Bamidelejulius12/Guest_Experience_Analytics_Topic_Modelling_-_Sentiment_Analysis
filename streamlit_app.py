import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Montra Guest Experience Analytics", page_icon=None, layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #f5f7fa; }
[data-testid="stSidebar"] { background-color: #111827; }
[data-testid="stSidebar"] * { color: white !important; }
.main-title { font-size: 32px; font-weight: 700; color: #111827; margin-bottom: 4px; }
.subtitle { color: #6b7280; font-size: 15px; margin-bottom: 25px; }
.section-title { font-size: 21px; font-weight: 650; color: #111827; margin-top: 25px; margin-bottom: 15px; }
.metric-card { background: white; padding: 18px; border-radius: 10px; border: 1px solid #e5e7eb; }
.metric-title { color: #6b7280; font-size: 13px; }
.metric-value { color: #111827; font-size: 27px; font-weight: 700; margin-top: 5px; }
.positive { color: #16a34a; }
.negative { color: #dc2626; }
.neutral { color: #d97706; }
div.stButton > button { border-radius: 7px; min-height: 44px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Single Prediction"

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "batch_file_name" not in st.session_state:
    st.session_state.batch_file_name = None


def metric_card(title, value, css_class=""):
    st.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value {css_class}">{value}</div></div>', unsafe_allow_html=True)


def check_api():
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def single_prediction(review):
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": review}, timeout=60)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API returned status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def batch_prediction(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(f"{API_URL}/predict/batch", files=files, timeout=1800)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                return pd.DataFrame(result)
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
                return None
        st.error(f"Batch prediction failed with status code {response.status_code}")
        return None
    except requests.exceptions.Timeout:
        st.error("The batch prediction timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API: {e}")
        return None


def clean_batch_data(df):
    df = df.copy()
    if "rating_score" in df.columns:
        df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
    if "sentiment_confidence" in df.columns:
        df["sentiment_confidence"] = pd.to_numeric(df["sentiment_confidence"], errors="coerce")
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df


with st.sidebar:
    st.markdown("<h2>MONTRA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af;'>Guest Experience Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Dashboard")

    if st.button("Single Prediction", use_container_width=True):
        st.session_state.page = "Single Prediction"

    if st.button("Batch Analysis", use_container_width=True):
        st.session_state.page = "Batch Analysis"

    if st.button("Model Training", use_container_width=True):
        st.session_state.page = "Model Training"

    if st.button("API Status", use_container_width=True):
        st.session_state.page = "API Status"

    st.markdown("---")
    st.caption("Montra Guest Experience Analytics")


# ============================================================
# SINGLE PREDICTION
# ============================================================

if st.session_state.page == "Single Prediction":

    st.markdown('<div class="main-title">Guest Review Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyse an individual guest review using the Montra sentiment model.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Guest Information</div>', unsafe_allow_html=True)
        hotel_name = st.text_input("Hotel Name", placeholder="Enter hotel name")
        rating = st.slider("Guest Rating", 1, 5, 5)
        room_type = st.selectbox("Room Type", ["Standard", "Deluxe", "Suite", "Executive", "Family"])
        booking_channel = st.selectbox("Booking Channel", ["Direct Website", "Booking.com", "Travel Agent", "Other"])

    with right:
        st.markdown('<div class="section-title">Guest Feedback</div>', unsafe_allow_html=True)
        feedback = st.text_area("Feedback", height=220, placeholder="Enter the guest's feedback here...")
        st.caption("Only the feedback text is sent to the sentiment model.")
        analyse = st.button("Analyse Review", type="primary", use_container_width=True)

    if analyse:
        if not feedback.strip():
            st.warning("Please enter the guest feedback.")
        else:
            with st.spinner("Analysing review..."):
                result = single_prediction(feedback)

            if "error" in result:
                st.error(result["error"])
            else:
                label = str(result.get("label", "Unknown"))
                confidence = float(result.get("confidence", 0))
                sentiment_class = "positive" if label.lower() == "positive" else "negative" if label.lower() == "negative" else "neutral"

                st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    metric_card("Hotel", hotel_name if hotel_name else "Not provided")

                with col2:
                    metric_card("Sentiment", label.upper(), sentiment_class)

                with col3:
                    metric_card("Confidence", f"{confidence * 100:.1f}%")


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif st.session_state.page == "Batch Analysis":

    st.markdown('<div class="main-title">Batch Guest Experience Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Run sentiment analysis across your hotel review dataset and evaluate individual hotel performance.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Review Dataset", type=["csv"])

    if uploaded_file is not None:

        preview_df = pd.read_csv(uploaded_file)

        required_columns = ["review_text", "hotel_name", "rating_score"]

        missing_columns = [column for column in required_columns if column not in preview_df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:

            st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
            st.dataframe(preview_df.head(5), use_container_width=True, hide_index=True)

            if st.session_state.batch_file_name != uploaded_file.name:
                st.session_state.batch_results = None
                st.session_state.batch_file_name = uploaded_file.name

            if st.button("Run Batch Prediction", type="primary", use_container_width=True):

                with st.spinner("Running sentiment analysis on the dataset..."):
                    result_df = batch_prediction(uploaded_file)

                if result_df is not None:
                    result_df = clean_batch_data(result_df)
                    st.session_state.batch_results = result_df
                    st.success(f"Batch prediction completed for {len(result_df):,} reviews.")

    if st.session_state.batch_results is not None:

        df = st.session_state.batch_results

        st.markdown("---")
        st.markdown('<div class="section-title">Overall Performance</div>', unsafe_allow_html=True)

        total_reviews = len(df)
        positive_count = df["sentiment_label"].astype(str).str.lower().eq("positive").sum()
        neutral_count = df["sentiment_label"].astype(str).str.lower().eq("neutral").sum()
        negative_count = df["sentiment_label"].astype(str).str.lower().eq("negative").sum()
        average_rating = df["rating_score"].mean()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            metric_card("Total Reviews", f"{total_reviews:,}")

        with col2:
            metric_card("Average Rating", f"{average_rating:.2f} / 5")

        with col3:
            metric_card("Positive Reviews", f"{positive_count:,}", "positive")

        with col4:
            metric_card("Neutral Reviews", f"{neutral_count:,}", "neutral")

        with col5:
            metric_card("Negative Reviews", f"{negative_count:,}", "negative")

        st.markdown('<div class="section-title">Overall Sentiment</div>', unsafe_allow_html=True)

        sentiment_df = df["sentiment_label"].astype(str).str.lower().value_counts().reset_index()
        sentiment_df.columns = ["sentiment", "count"]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(sentiment_df, names="sentiment", values="count", hole=0.55, title="Sentiment Distribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(sentiment_df, x="sentiment", y="count", text="count", title="Number of Reviews by Sentiment")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Hotel Performance</div>', unsafe_allow_html=True)

        hotel_list = sorted(df["hotel_name"].dropna().astype(str).unique().tolist())

        selected_hotel = st.selectbox("Select Hotel", hotel_list)

        hotel_df = df[df["hotel_name"].astype(str) == selected_hotel].copy()

        hotel_reviews = len(hotel_df)
        hotel_rating = hotel_df["rating_score"].mean()
        hotel_positive = hotel_df["sentiment_label"].astype(str).str.lower().eq("positive").sum()
        hotel_neutral = hotel_df["sentiment_label"].astype(str).str.lower().eq("neutral").sum()
        hotel_negative = hotel_df["sentiment_label"].astype(str).str.lower().eq("negative").sum()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            metric_card("Total Reviews", f"{hotel_reviews:,}")

        with col2:
            metric_card("Average Rating", f"{hotel_rating:.2f} / 5")

        with col3:
            metric_card("Positive", f"{hotel_positive:,}", "positive")

        with col4:
            metric_card("Neutral", f"{hotel_neutral:,}", "neutral")

        with col5:
            metric_card("Negative", f"{hotel_negative:,}", "negative")

        st.markdown(f'<div class="section-title">{selected_hotel} Performance</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        hotel_sentiment = hotel_df["sentiment_label"].astype(str).str.lower().value_counts().reset_index()
        hotel_sentiment.columns = ["sentiment", "count"]

        with col1:
            fig = px.pie(hotel_sentiment, names="sentiment", values="count", hole=0.55, title="Sentiment Distribution")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            rating_df = hotel_df["rating_score"].value_counts().sort_index().reset_index()
            rating_df.columns = ["rating", "count"]
            fig = px.bar(rating_df, x="rating", y="count", text="count", title="Rating Distribution")
            fig.update_layout(height=420, xaxis_title="Rating Score", yaxis_title="Number of Reviews")
            st.plotly_chart(fig, use_container_width=True)

        rating_sentiment = hotel_df.groupby(["rating_score", "sentiment_label"]).size().reset_index(name="count")

        fig = px.bar(rating_sentiment, x="rating_score", y="count", color="sentiment_label", barmode="group", title="Rating Score vs Sentiment")
        fig.update_layout(height=420, xaxis_title="Rating Score", yaxis_title="Number of Reviews")
        st.plotly_chart(fig, use_container_width=True)

        if "review_date" in hotel_df.columns:

            trend_df = hotel_df.dropna(subset=["review_date"]).groupby(pd.Grouper(key="review_date", freq="ME")).agg(average_rating=("rating_score", "mean"), reviews=("review_id", "count")).reset_index()

            if not trend_df.empty:
                fig = px.line(trend_df, x="review_date", y="average_rating", markers=True, title="Average Rating Over Time")
                fig.update_layout(height=400, xaxis_title="Review Date", yaxis_title="Average Rating")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Hotel Comparison</div>', unsafe_allow_html=True)

        hotel_summary = df.groupby("hotel_name").agg(total_reviews=("review_id", "count"), average_rating=("rating_score", "mean")).reset_index()

        positive_rates = df.assign(is_positive=df["sentiment_label"].astype(str).str.lower().eq("positive")).groupby("hotel_name")["is_positive"].mean().mul(100).reset_index(name="positive_percentage")

        negative_rates = df.assign(is_negative=df["sentiment_label"].astype(str).str.lower().eq("negative")).groupby("hotel_name")["is_negative"].mean().mul(100).reset_index(name="negative_percentage")

        hotel_summary = hotel_summary.merge(positive_rates, on="hotel_name", how="left")
        hotel_summary = hotel_summary.merge(negative_rates, on="hotel_name", how="left")

        col1, col2 = st.columns(2)

        with col1:
            comparison_df = hotel_summary.sort_values("average_rating", ascending=True)
            fig = px.bar(comparison_df, x="average_rating", y="hotel_name", orientation="h", text=comparison_df["average_rating"].round(2), title="Average Rating by Hotel")
            fig.update_layout(height=max(400, len(comparison_df) * 35), xaxis_title="Average Rating", yaxis_title="Hotel")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            comparison_df = hotel_summary.sort_values("positive_percentage", ascending=True)
            fig = px.bar(comparison_df, x=["positive_percentage", "negative_percentage"], y="hotel_name", orientation="h", barmode="group", title="Positive vs Negative Reviews")
            fig.update_layout(height=max(400, len(comparison_df) * 35), xaxis_title="Percentage", yaxis_title="Hotel")
            st.plotly_chart(fig, use_container_width=True)

        hotel_summary["average_rating"] = hotel_summary["average_rating"].round(2)
        hotel_summary["positive_percentage"] = hotel_summary["positive_percentage"].round(1)
        hotel_summary["negative_percentage"] = hotel_summary["negative_percentage"].round(1)

        hotel_summary.columns = ["Hotel", "Reviews", "Average Rating", "Positive %", "Negative %"]

        st.dataframe(hotel_summary.sort_values("Average Rating", ascending=False), use_container_width=True, hide_index=True)

        st.download_button("Download Batch Predictions", data=df.to_csv(index=False).encode("utf-8"), file_name="montra_batch_predictions.csv", mime="text/csv", use_container_width=True)


# ============================================================
# MODEL TRAINING
# ============================================================

elif st.session_state.page == "Model Training":

    st.markdown('<div class="main-title">Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Train the Montra sentiment analysis model through the FastAPI service.</div>', unsafe_allow_html=True)

    st.warning("Training the model may take some time.")

    if st.button("Train Sentiment Model", type="primary", use_container_width=True):

        with st.spinner("Training model..."):

            try:
                response = requests.get(f"{API_URL}/train", timeout=1800)

                if response.status_code == 200:
                    result = response.json()

                    if result.get("status") == "success":
                        st.success(result.get("message", "Model training completed successfully."))
                    else:
                        st.error(result.get("message", "Model training failed."))

                else:
                    st.error(f"Training API returned status code {response.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to API: {e}")


# ============================================================
# API STATUS
# ============================================================

elif st.session_state.page == "API Status":

    st.markdown('<div class="main-title">API Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Check the connection between the dashboard and the FastAPI service.</div>', unsafe_allow_html=True)

    online = check_api()

    col1, col2 = st.columns(2)

    with col1:
        metric_card("API Status", "ONLINE" if online else "OFFLINE", "positive" if online else "negative")

    with col2:
        metric_card("API Endpoint", API_URL)

    st.markdown('<div class="section-title">Available Endpoints</div>', unsafe_allow_html=True)

    endpoint_df = pd.DataFrame({"Service": ["Single Prediction", "Batch Prediction", "Model Training"], "Endpoint": ["POST /predict", "POST /predict/batch", "GET /train"], "Purpose": ["Analyse one guest review", "Analyse an entire review dataset", "Train the sentiment model"]})

    st.dataframe(endpoint_df, use_container_width=True, hide_index=True)

