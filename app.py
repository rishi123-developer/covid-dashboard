# ------------------------- Imports & Global Setup -------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import plotly.express as px
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import av
import time
import re
from streamlit_lottie import st_lottie
import requests
import json

st.set_page_config(page_title="India COVID Dashboard", layout="wide")

# 🌒 Apply Dark Mode
st.markdown("""
<style>
    body { background-color: #121212; color: #e0e0e0; }
    .stButton>button, .stDownloadButton>button { background-color: #444; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align:center; padding:15px; background:linear-gradient(to right,#121212,#1f1f1f); border-radius:10px;">
        <h1 style="color:white;">🧠 India COVID-19 Dashboard</h1>
        <p style="color:#aaaaaa;">Voice-enabled | Data-driven | Designed by Rishi</p>
    </div>
""", unsafe_allow_html=True)
st.set_page_config(page_title="Rishi's Dashboard", layout="wide")

st.markdown("""
<marquee style='font-size:22px; color:#FFA07A; font-weight:bold;'>
🚨 Welcome to the India COVID-19 Dashboard – Stay Informed, Stay Safe 🙏
</marquee>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import plotly.express as px
# (Other imports...)

# ✅ App config and mobile CSS
st.set_page_config(page_title="Rishi's Dashboard", layout="wide")

# ✅ Responsive tweaks for mobile
st.markdown("""
    <style>
    @media only screen and (max-width: 600px) {
        .block-container {
            padding: 1rem !important;
        }
        h1, h2, h3 {
            font-size: 1.2rem !important;
        }
        .neon-btn {
            padding: 8px 16px !important;
            font-size: 14px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

mic_lottie = load_lottieurl("https://lottie.host/7149385e-bb28-42f7-bf97-dc10d6265e0d/iQvnrRjEPb.json")
with open("covid_animation.json", "r") as f:
    lottie_covid = json.load(f)
    st_lottie(lottie_covid, height=200)



# ------------------------- Load Datasets -------------------------
@st.cache_data
def load_data():
    df_summary = pd.read_csv("updated_final_cleaned_covid_data.csv")
    df_static = pd.read_csv("cleaned_covid_19_data_2020_2024.csv", on_bad_lines="skip")
    df_ml = pd.read_csv("covid_ml_ready.csv")
    df_vax = pd.read_csv("covid_vaccine_statewise.csv")
    df_test = pd.read_csv("StatewiseTestingDetails.csv")
    return df_summary, df_static, df_ml, df_vax, df_test

df_summary, df_static, df_ml, df_vax, df_test = load_data()

# Manual CSV Load
try:
    df_manual = pd.read_csv("covid_19 data 2020 to 2023.csv")
    df_manual.columns = df_manual.columns.str.strip().str.lower()
    if "date" not in df_manual.columns:
        df_manual = pd.DataFrame(columns=["date", "state", "confirmed", "deaths"])
    else:
        df_manual["date"] = pd.to_datetime(df_manual["date"], errors="coerce")
        df_manual = df_manual.dropna(subset=["date"]).sort_values("date")
except FileNotFoundError:
    df_manual = pd.DataFrame(columns=["date", "state", "confirmed", "deaths"])

# Preprocessing Function
def preprocess(df, date_col=None, state_col="state"):
    df.columns = df.columns.str.strip().str.lower()
    if state_col in df.columns:
        df[state_col] = df[state_col].astype(str).str.strip().str.lower()
    if date_col and date_col.lower() in df.columns:
        df[date_col.lower()] = pd.to_datetime(df[date_col.lower()], errors="coerce")
        df.dropna(subset=[date_col.lower()], inplace=True)
    return df

df_summary = preprocess(df_summary, "date")
df_ml = preprocess(df_ml, "date")
df_test = preprocess(df_test, "date")
df_vax = preprocess(df_vax, "updated on")
df_manual = preprocess(df_manual, "date")


df_summary["date"] = pd.to_datetime(df_summary["date"], errors="coerce")
df_vax["updated on"] = pd.to_datetime(df_vax["updated on"], errors="coerce")
df_test["date"] = pd.to_datetime(df_test["date"], errors="coerce")
df_ml["date"] = pd.to_datetime(df_ml["date"], errors="coerce")


import streamlit as st
import json
from datetime import datetime
from streamlit_lottie import st_lottie

def Sidebar(df_summary):
    with st.sidebar:
        # --- Branding Header ---
        st.markdown("""
            <div style="text-align:center; margin-bottom:15px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Flag_of_India.svg/64px-Flag_of_India.svg.png" width="60">
                <h3 style="color:#ffffff; margin-top:10px;">Rishi's Dashboard</h3>
                <p style="color:#aaaaaa; font-size:13px;">Live · Voice · Visual</p>
                <hr style="border: 0.5px solid #444444;">
            </div>
        """, unsafe_allow_html=True)

        # --- Version & Time ---
        st.markdown(f"""
            <div style="text-align:center; margin-bottom:15px;">
                <span style="color:#00FFAB; font-size:13px;">
                🛠️ Version: <b>v1.2</b><br>
                🗓️ Updated: <b>July 2025</b><br>
                ⏱️ Loaded at: <b>{datetime.now().strftime('%H:%M:%S')}</b>
                </span>
            </div>
        """, unsafe_allow_html=True)

        # --- Navigation Title ---
        st.title("🧭 COVID Dashboard")
        if st.button("🔄 Refresh", key="neon_refresh_dashboard"):
            time.sleep(0.5)
            st.rerun()


        # --- Navigation Tabs ---
        tabs = [
            "📊 Summary", "💉 Vaccination", "🧪 Testing",
            "🤖 ML Prediction", "🏆 Top States", "✏️ Manual Entry"
        ]
        tab = st.radio("Choose View", tabs)

        # --- State Selector ---
        state_options = sorted(df_summary["state"].dropna().unique())
        selected_state = st.selectbox("Select a State", state_options)

        # --- Live Summary ---
        st.markdown("### 🧮 Live Summary")
        st.metric("Total Confirmed", int(df_summary["confirmed"].sum()))
        st.metric("Total Recovered", int(df_summary["recovered"].sum()))
        st.metric("Total Deaths", int(df_summary["deaths"].sum()))

        # --- Voice Status & Toggle ---
        if st.session_state.get("trigger_voice", False):
            st.success("🎙️ Voice Listening Active")
        else:
            st.info("🔇 Voice Input Off")

        st.checkbox("🎧 Enable Voice Mode", key="voice_mode_toggle")

        # --- Optional Animation (sidebar_animation.json required) ---
        try:
            with open("sidebar_animation.json", "r") as f:
                sidebar_anim = json.load(f)
            st_lottie(sidebar_anim, height=100, speed=1.2, key="side_anim")
        except Exception as e:
            st.warning("🎬 No animation loaded")

        # --- Tools List ---
        with st.expander("📁 Dashboard Options"):
            st.markdown("""
                - ✔️ Manual Data Entry  
                - 🎙️ Voice Input Mode  
                - 📊 Charts & Summary  
                - ✅ Save Success Animation  
            """)

    return tab, selected_state

tab, selected_state = Sidebar(df_summary)
if tab == "📊 Summary":
    st.title(f"📊 COVID Summary – {selected_state.title()}")

    # 🗂️ Date parsing
    df_summary["date"] = pd.to_datetime(df_summary["date"], errors="coerce")

    # 🔍 Filter selected state
    df_state = df_summary[df_summary["state"].str.lower() == selected_state.lower()].copy()

    # 🧮 Latest metrics
    latest = df_state.sort_values("date", ascending=False).head(1)
    if not latest.empty:
        st.metric("Confirmed", int(latest["confirmed"].values[0]))
        st.metric("Recovered", int(latest["recovered"].values[0]))
        st.metric("Deaths", int(latest["deaths"].values[0]))
    else:
        st.warning("⚠️ No data found for selected state.")

    # 📆 Date range selector
    min_date, max_date = df_state["date"].min(), df_state["date"].max()
    start_date, end_date = st.date_input("Select date range:", [min_date, max_date])

    # 🔁 View mode toggle
    view_mode = st.radio("View Mode", ["Cumulative", "Daily"], horizontal=True)

    # 📄 Filter by date range
    df_filtered = df_state[
        (df_state["date"] >= pd.to_datetime(start_date)) &
        (df_state["date"] <= pd.to_datetime(end_date))
    ].sort_values("date").copy()

    # 🔢 Compute daily values if needed
    if view_mode == "Daily":
        df_filtered[["confirmed", "recovered", "deaths"]] = (
            df_filtered[["confirmed", "recovered", "deaths"]]
            .diff()
            .fillna(0)
            .astype(int)
        )

    # 📋 Show filtered data
    st.dataframe(df_filtered.reset_index(drop=True))

    # 📈 Plot chart
    fig = px.line(
        df_filtered,
        x="date",
        y=["recovered", "deaths"],
        markers=True,
        title="Recovery and Death Trends",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True, key="summary_chart")

    # 🔮 ML Forecast (Confirmed)
    if "confirmed" in df_filtered.columns:
        df_forecast = df_filtered[["date", "confirmed"]].dropna()
        st.write("🧮 Forecast Data Points:", len(df_forecast))
        st.write(df_forecast.tail())  # Optional: peek at data
            
        if len(df_forecast) >= 10:
            st.subheader("🔮 Confirmed Cases Forecast (Next 7 Days)")
                
            df_forecast["date"] = pd.to_datetime(df_forecast["date"], errors="coerce")
            df_forecast["days"] = (df_forecast["date"] - df_forecast["date"].min()).dt.days
            X = df_forecast[["days"]]
            y = df_forecast["confirmed"]

            try:
                model = LinearRegression().fit(X, y)

                future_days = np.arange(X["days"].max() + 1, X["days"].max() + 8).reshape(-1, 1)
                future_preds = model.predict(future_days)
                future_dates = pd.date_range(start=df_forecast["date"].max() + pd.Timedelta(days=1), periods=7)

                df_future = pd.DataFrame({
                    "date": future_dates,
                    "Predicted Confirmed": np.maximum(future_preds, 0).astype(int)
                })

                fig_pred = px.line(
                    df_future,
                    x="date",
                    y="Predicted Confirmed",
                    markers=True,
                    line_dash_sequence=["dash"],
                    title="Predicted Confirmed Cases",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            except Exception as e:
                st.error(f"Model training failed: {e}")
        else:
            st.info(f"ℹ️ Only {len(df_forecast)} data points available. Forecast needs at least 10 non-null rows.")
    else:
        st.warning("⚠️ 'confirmed' column not available in filtered dataset.")

    # 📥 Download CSV
    st.download_button(
        label="📥 Download Filtered CSV",
        data=df_filtered.to_csv(index=False),
        file_name="summary_filtered.csv",
        mime="text/csv"
    )
    st.dataframe(df_filtered.sort_values("date", ascending=False).head(10))
    if st.button("🔄 Refresh", key="neon_refresh_summary"):
        st.rerun()



elif tab == "💉 Vaccination":
    st.title(f"💉 Vaccination Progress – {selected_state.title()}")

    state_df = df_summary[df_summary["state"] == selected_state]

    with st.sidebar:
        st.markdown(f"### 📍 {selected_state.title()} Summary")
        st.metric("Confirmed", int(state_df["confirmed"].sum()))
        st.metric("Recovered", int(state_df["recovered"].sum()))
        st.metric("Deaths", int(state_df["deaths"].sum()))

    # 👇 Ensure proper case match (like summary tab)
    df_v = df_vax[df_vax["state"].str.lower() == selected_state.lower()]

    # 👇 Ensure date column is in datetime format
    df_v["updated on"] = pd.to_datetime(df_v["updated on"], errors="coerce")

    if df_v.empty or "total individuals vaccinated" not in df_v.columns:
        st.warning("No vaccination data available for this state.")
    else:
        min_date = df_v["updated on"].min()
        max_date = df_v["updated on"].max()

        start_date, end_date = st.date_input(
            "Select date range:",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="vax_date"
        )

        df_v_filtered = df_v[
            (df_v["updated on"] >= pd.to_datetime(start_date)) &
            (df_v["updated on"] <= pd.to_datetime(end_date))
        ].copy()

        # 📈 Plotly Area Chart
        fig_vax = px.area(
            df_v_filtered,
            x="updated on",
            y="total individuals vaccinated",
            title="Total Individuals Vaccinated Over Time",
            template="plotly_dark",
            markers=True
        )
        st.plotly_chart(fig_vax, use_container_width=True)

        # 📥 CSV Export
        st.download_button(
            label="📥 Download Vaccination Data",
            data=df_v_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_state}_vaccination_data.csv",
            mime="text/csv"
        )

        st.dataframe(df_v_filtered.tail(10))
        if st.button("🔄 Refresh", key="neon_refresh_vaccination"):
            st.rerun()


elif tab == "🧪 Testing":
    st.title(f"🧪 COVID Testing – {selected_state.title()}")

    state_df = df_summary[df_summary["state"] == selected_state]

    with st.sidebar:
        st.markdown(f"### 📍 {selected_state.title()} Summary")
        st.metric("Confirmed", int(state_df["confirmed"].sum()))
        st.metric("Recovered", int(state_df["recovered"].sum()))
        st.metric("Deaths", int(state_df["deaths"].sum()))

    # 🔍 Standardize state column for safety
    df_t = df_test[df_test["state"].str.lower() == selected_state.lower()].copy()

    # ✅ Ensure date column is datetime (avoids '<' TypeError)
    df_t["date"] = pd.to_datetime(df_t["date"], errors="coerce")

    if df_t.empty or "totalsamples" not in df_t.columns:
        st.warning("No testing data available for this state.")
    else:
        min_date = df_t["date"].min()
        max_date = df_t["date"].max()

        start_date, end_date = st.date_input(
            "Select date range:",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="test_date"
        )

        view_mode = st.radio(
            "View Mode", ["Cumulative", "Daily"],
            horizontal=True,
            key="test_mode"
        )

        df_t_filtered = df_t[
            (df_t["date"] >= pd.to_datetime(start_date)) &
            (df_t["date"] <= pd.to_datetime(end_date))
        ].copy()

        if view_mode == "Daily":
            df_t_filtered["totalsamples"] = df_t_filtered["totalsamples"].diff().fillna(0)

        # 📈 Plotly Chart
        fig_test = px.line(
            df_t_filtered,
            x="date",
            y="totalsamples",
            title="COVID Tests Over Time",
            markers=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig_test, use_container_width=True)

        # 📥 CSV Export
        st.download_button(
            label="📥 Download Testing Data",
            data=df_t_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_state}_testing_data.csv",
            mime="text/csv"
        )

        st.dataframe(df_t_filtered.tail(10))
        if st.button("🔄 Refresh", key="neon_refresh_testing"):
            st.rerun()


        # ------------------------- 🤖 ML Prediction View -------------------------
elif tab == "🤖 ML Prediction":
    st.title(f"🤖 ML-Based Timeline – {selected_state.title()}")

    state_df = df_summary[df_summary["state"] == selected_state]

    with st.sidebar:
        st.markdown(f"### 📍 {selected_state.title()} Summary")
        st.metric("Confirmed", int(state_df["confirmed"].sum()))
        st.metric("Recovered", int(state_df["recovered"].sum()))
        st.metric("Deaths", int(state_df["deaths"].sum()))

    # 📍 Match state case-insensitively
    df_m = df_ml[df_ml["state"].str.lower() == selected_state.lower()].copy()

    # ✅ Convert date column safely
    df_m["date"] = pd.to_datetime(df_m["date"], errors="coerce")

    if df_m.empty or not all(col in df_m.columns for col in ["confirmed", "recovered", "deaths"]):
        st.warning("ML data missing or incomplete.")
    else:
        min_date = df_m["date"].min()
        max_date = df_m["date"].max()
        start_date, end_date = st.date_input("Date range:", [min_date, max_date], key="ml_date")

        view_mode = st.radio("View Mode", ["Cumulative", "Daily"], horizontal=True, key="ml_toggle")
        df_filtered = df_m[
            (df_m["date"] >= pd.to_datetime(start_date)) &
            (df_m["date"] <= pd.to_datetime(end_date))
        ].copy()

        if view_mode == "Daily":
            df_filtered[["confirmed", "recovered", "deaths"]] = (
                df_filtered[["confirmed", "recovered", "deaths"]].diff().fillna(0)
            )

        # 📈 Plot ML Trends
        fig_ml = px.line(
            df_filtered,
            x="date",
            y=["confirmed", "recovered", "deaths"],
            title="ML COVID Trends",
            markers=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        # 🔮 Forecast Section
        st.subheader("🔮 Forecast: Confirmed Cases (Next 7 Days)")

        df_forecast = df_filtered[["date", "confirmed"]].dropna()
        df_forecast["date"] = pd.to_datetime(df_forecast["date"], errors="coerce")
        df_forecast["days"] = (df_forecast["date"] - df_forecast["date"].min()).dt.days

        if len(df_forecast) >= 10:
            X = df_forecast[["days"]]
            y = df_forecast["confirmed"]

            model = LinearRegression().fit(X, y)
            future_days = np.arange(X["days"].max() + 1, X["days"].max() + 8).reshape(-1, 1)
            preds = model.predict(future_days)
            future_dates = pd.date_range(start=df_forecast["date"].max() + pd.Timedelta(days=1), periods=7)

            df_pred = pd.DataFrame({
                "date": future_dates,
                "Predicted Confirmed": np.maximum(preds, 0).astype(int)
            })

            fig_pred = px.line(
                df_pred,
                x="date",
                y="Predicted Confirmed",
                title="Predicted Confirmed Cases",
                markers=True,
                template="plotly_dark"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("Need at least 10 data points for forecasting.")

        # 📥 Export
        st.download_button(
            "📥 Download ML Data",
            data=df_filtered.to_csv(index=False),
            file_name=f"{selected_state}_ml_data.csv"
        )

        if st.checkbox("Show raw ML data"):
            st.dataframe(df_filtered.tail(10))
        if st.button("🔄 Refresh", key="neon_refresh_ml"):
            st.rerun()


            # ------------------------- 🏆 Top States Comparison -------------------------
elif tab == "🏆 Top States":
    st.title("🏆 Top 5 States by COVID Trends")

    # ✅ Confirmed Growth (last 7 days)
    st.subheader("📈 States with Highest Growth (Last 7 Days)")

    if "confirmed" not in df_summary.columns:
        st.warning("⚠️ 'confirmed' column missing in df_summary.")
    else:
        df_summary["date"] = pd.to_datetime(df_summary["date"], errors="coerce")
        df_recent = df_summary[df_summary["date"] >= df_summary["date"].max() - pd.Timedelta(days=7)]

        if not df_recent.empty:
            top_growth = (
                df_recent.groupby("state", as_index=False)["confirmed"]
                .sum(numeric_only=True)
                .sort_values(by="confirmed", ascending=False)
                .set_index("state")
                .head(5)
            )

            fig_growth = px.bar(
                top_growth[::-1],
                orientation="h",
                title="Top 5 States by Case Growth (Last 7 Days)",
                labels={"confirmed": "Confirmed Cases", "state": "State"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.warning("⚠️ No recent COVID growth data available.")

    # 💉 Vaccination Coverage
    st.subheader("💉 States with Highest Vaccination")

    if "total individuals vaccinated" not in df_vax.columns or df_vax.empty:
        st.warning("⚠️ Vaccination data unavailable or column missing.")
    else:
        df_vax["updated on"] = pd.to_datetime(df_vax["updated on"], errors="coerce")
        latest_vax = df_vax.sort_values("updated on").dropna(subset=["total individuals vaccinated"])
        latest_vax = latest_vax.groupby("state", as_index=False).last()

        top_vax = (
            latest_vax[["state", "total individuals vaccinated"]]
            .sort_values(by="total individuals vaccinated", ascending=False)
            .head(5)
            .set_index("state")
        )

        fig_vax = px.bar(
            top_vax[::-1],
            orientation="h",
            title="Top 5 States by Vaccination",
            labels={"total individuals vaccinated": "People Vaccinated"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_vax, use_container_width=True)
        st.dataframe(top_vax[::-1])
        if st.button("🔄 Refresh", key="neon_refresh_top_5_state"):
            st.rerun()


        # ------------------------- ✏️ Manual COVID Data Entry -------------------------
    # --- Helper: voice to state mapping ---
    def extract_state_name(text, state_list):
        for s in state_list:
            if s.lower() in text.lower():
                return s
        return None

    # --- Helper: extract numbers ---
    def extract_numbers(text):
        return [int(num) for num in re.findall(r'\d+', text)]

# --- Manual Entry Tab ---
elif tab == "✏️ Manual Entry":
    st.title("✏️ Manual COVID-19 Data Entry")

    file_summary = "updated_final_cleaned_covid_data.csv"
    backup_summary = "final_cleaned_covid_data.csv"

    if os.path.exists(file_summary):
        df_summary = pd.read_csv(file_summary)
    elif os.path.exists(backup_summary):
        df_summary = pd.read_csv(backup_summary)
    else:
        df_summary = pd.DataFrame(columns=["date", "state", "confirmed", "recovered", "deaths"])

    df_summary["date"] = pd.to_datetime(df_summary["date"], errors="coerce")
    df_summary["state"] = df_summary["state"].astype(str).str.strip().str.lower()

    all_states = sorted(df_summary["state"].dropna().unique().tolist())
    state_input = st.selectbox("Select State", all_states)

    filtered = df_summary[df_summary["state"] == state_input]
    latest_entries = filtered.sort_values("date", ascending=False).head(10)

    selected_entry_date = st.date_input(
        "Select date to add/edit entry:",
        value=pd.to_datetime(latest_entries["date"].max()) if not latest_entries.empty else pd.Timestamp.today(),
        min_value=filtered["date"].min() if not filtered.empty else pd.to_datetime("2020-01-01"),
        max_value=pd.Timestamp.today()
    ).strftime("%Y-%m-%d")

    if selected_entry_date in latest_entries["date"].dt.strftime("%Y-%m-%d").tolist():
        row = latest_entries[latest_entries["date"].dt.strftime("%Y-%m-%d") == selected_entry_date].iloc[0]
        default_cases = int(row["confirmed"])
        default_recovered = int(row["recovered"])
        default_deaths = int(row["deaths"])
    else:
        default_cases = default_recovered = default_deaths = 0

   # --- Voice Fill (Final Stable Version) ---
    st.subheader("🎙️ Voice Fill (Optional)")

    if "trigger_voice" not in st.session_state:
        st.session_state["trigger_voice"] = False

    # ✅ Load animation file properly
    try:
        with open("mic_animation.json", "r") as f:
            mic_lottie = json.load(f)
    except Exception as e:
        mic_lottie = None
        st.warning(f"⚠️ Couldn't load mic animation: {e}")

    # ✅ Show animation container
    st.markdown("""
    <div style="background-color:#1c1c1c; padding:15px; border-radius:10px; text-align:center;">
        <h4 style="color:white;">🎧 Speak to Enter</h4>
    </div>
    """, unsafe_allow_html=True)

    if mic_lottie:
        st_lottie(mic_lottie, speed=1.2, height=160, key="mic_input")

    # ✅ Button to start stream
    if st.button("🎧 Speak Now"):
        st.session_state["trigger_voice"] = True

    # ✅ Activate mic stream
    if st.session_state["trigger_voice"]:
        ctx = webrtc_streamer(
            key="voice-entry",
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )

        if ctx and ctx.state.playing:
            st_lottie(mic_lottie, speed=1, height=100, key="mic_on")
            import time
            time.sleep(1.5)  # Let the mic get ready

            import speech_recognition as sr
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    st.markdown("🟢 **Listening...** Speak like: _‘Gujarat 120 confirmed 80 recovered 5 deaths save entry’_")
                    audio = r.listen(source, timeout=2, phrase_time_limit=6)
                    result = r.recognize_google(audio).lower()
                    st.success(f"🗣️ Heard: `{result}`")

                    # Helpers you already have
                    numbers = extract_numbers(result)
                    spoken_state = extract_state_name(result, all_states)

                    if spoken_state:
                        state_input = spoken_state
                    if len(numbers) > 0: default_cases = numbers[0]
                    if len(numbers) > 1: default_recovered = numbers[1]
                    if len(numbers) > 2: default_deaths = numbers[2]

                    st.session_state["trigger_save_voice"] = "save entry" in result

            except sr.UnknownValueError:
                st.warning("😕 Could not understand your voice.")
            except Exception as e:
                st.error(f"🚫 Mic Error: {e}")

            st.session_state["trigger_voice"] = False  # Reset after capture
        
    # --- Form Inputs ---
    confirmed = st.number_input("Confirmed cases:", min_value=0, value=default_cases, step=1)
    recovered = st.number_input("Recovered cases:", min_value=0, value=default_recovered, step=1)
    deaths = st.number_input("Deaths:", min_value=0, value=default_deaths, step=1)

    # --- Save via Button or Voice Trigger ---
    should_save = st.button("💾 Save Entry") or st.session_state.get("trigger_save_voice", False)

    if should_save:
        try:
            # 📆 Prepare new entry
            new_date = pd.to_datetime(selected_entry_date)
            new_row = pd.DataFrame([{
                "date": new_date,
                "state": state_input,
                "confirmed": confirmed,
                "recovered": recovered,
                "deaths": deaths
            }])

            # 🔄 Remove old entry for same state-date if exists
            df_summary = df_summary[~((df_summary["state"] == state_input) & (df_summary["date"] == new_date))]
            df_summary = pd.concat([df_summary, new_row], ignore_index=True)
            df_summary.drop_duplicates(subset=["state", "date"], keep="last", inplace=True)
            df_summary.to_csv(file_summary, index=False)

            # ✅ Show success message
            st.success(f"✅ Entry saved for {state_input.title()} on {selected_entry_date}")

            # 🎉 Load & show animation (from local .json file)
            with open("success_animation.json", "r") as f:
                success_anim = json.load(f)

            if success_anim:
                st_lottie(success_anim, height=180, speed=1.2, key="save_anim")

            # ✅ Optional: reset voice trigger
            st.session_state["trigger_save_voice"] = False

            # 🔁 Optional: rerun to refresh UI (only if needed)
            if st.button("🔄 Refresh", key="refresh_btn"):
                st.rerun()

        except Exception as e:
            st.error(f"🚫 Save failed: {e}")

    # --- Delete / Undo ---
    if selected_entry_date in latest_entries["date"].dt.strftime("%Y-%m-%d").tolist():
        if st.button("🗑️ Delete This Entry"):
            to_delete = (
                (df_summary["state"] == state_input) &
                (df_summary["date"].dt.strftime("%Y-%m-%d") == selected_entry_date)
            )
            st.session_state["deleted_entry"] = df_summary[to_delete].copy()
            df_summary = df_summary[~to_delete]
            df_summary.to_csv(file_summary, index=False)
            st.success(f"🗑️ Deleted entry for {state_input.title()} on {selected_entry_date}")
            st.rerun()

    if "deleted_entry" in st.session_state and not st.session_state["deleted_entry"].empty:
        if st.button("↩️ Undo Delete"):
            df_summary = pd.concat([df_summary, st.session_state["deleted_entry"]], ignore_index=True)
            df_summary.to_csv(file_summary, index=False)
            st.success("↩️ Entry restored.")
            st.session_state["deleted_entry"] = pd.DataFrame()
            
            import time
            time.sleep(1)  # slight delay before rerun
            st.rerun()

    # --- Preview & Chart ---
    st.subheader("📄 Last 10 Entries")
    st.dataframe(latest_entries.sort_values("date", ascending=False))

    # 📊 Chart Section with Styled Container
    st.subheader("📊 Chart View – Last 10 Entries")

    recent = df_summary[df_summary["state"] == state_input].sort_values("date", ascending=False).head(10)

    fig = px.bar(
        recent,
        x="date",
        y=["confirmed", "recovered", "deaths"],
        barmode="group",
        title=f"📊 Last 10 Days COVID Stats – {state_input.title()}",
        template="plotly_dark"
    )

    # 💡 Wrap inside a styled div and use unique key
    st.markdown("""
    <div style="background:#1a1d21; padding:20px; border-radius:10px; margin-top:20px;">
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True, key="manual_entry_chart")
    st.markdown("</div>", unsafe_allow_html=True)

    # ...Your app code: charts, summary, manual entry etc...
    if st.button("🔄 Refresh", key="neon_refresh_manually"):
        time.sleep(0.5)
        st.rerun()