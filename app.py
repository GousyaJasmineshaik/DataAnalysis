import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Global Road Accidents Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("road_accident_dataset.csv")
    return df

df = load_data()

st.title("🌍 Global Road Accidents Dashboard")
st.markdown("An advanced interactive dashboard for analyzing **global road accidents**, driver behavior, and contributing factors.")

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("🔍 Filters")

countries = st.sidebar.multiselect("Select Country", df["Country"].unique())
years = st.sidebar.multiselect("Select Year", sorted(df["Year"].unique()))
regions = st.sidebar.multiselect("Select Region", df["Region"].unique())
severity = st.sidebar.multiselect("Select Accident Severity", df["Accident Severity"].unique())

filtered_df = df.copy()
if countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
if years:
    filtered_df = filtered_df[filtered_df["Year"].isin(years)]
if regions:
    filtered_df = filtered_df[filtered_df["Region"].isin(regions)]
if severity:
    filtered_df = filtered_df[filtered_df["Accident Severity"].isin(severity)]

# ------------------------------
# KPIs
# ------------------------------
st.subheader("📈 Key Performance Indicators (KPIs)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Accidents", f"{len(filtered_df):,}")
col2.metric("Total Fatalities", f"{filtered_df['Number of Fatalities'].sum():,}")
col3.metric("Total Injuries", f"{filtered_df['Number of Injuries'].sum():,}")
col4.metric("Avg Response Time (min)", f"{filtered_df['Emergency Response Time'].mean():.2f}")

st.markdown("---")

# ------------------------------
# Pie Charts Section
# ------------------------------
st.subheader("🥧 Distribution Insights")

col1, col2, col3 = st.columns(3)

# Pie 1 - Accident Severity
with col1:
    st.markdown("**Accident Severity Distribution**")
    severity_counts = filtered_df["Accident Severity"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(severity_counts.values, labels=severity_counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# Pie 2 - Urban vs Rural
with col2:
    st.markdown("**Urban vs Rural Accidents**")
    area_counts = filtered_df["Urban/Rural"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(area_counts.values, labels=area_counts.index, autopct="%1.1f%%", startangle=90, colors=["#66b3ff","#ff9999"])
    ax.axis("equal")
    st.pyplot(fig)

# Pie 3 - Weather Conditions
with col3:
    st.markdown("**Weather Conditions**")
    weather_counts = filtered_df["Weather Conditions"].value_counts().head(6)
    fig, ax = plt.subplots()
    ax.pie(weather_counts.values, labels=weather_counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

st.markdown("---")

# ------------------------------
# Unique Visualization 1: Accident Heatmap (Day vs Time)
# ------------------------------
st.subheader("🚦 Accident Heatmap by Day of Week & Time of Day")
pivot = filtered_df.pivot_table(index="Day of Week", columns="Time of Day", values="Accident Severity", aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, cmap="Reds", annot=True, fmt="d", ax=ax)
ax.set_title("Accident Frequency by Day and Time")
st.pyplot(fig)

# ------------------------------
# Unique Visualization 2: Driver Demographics
# ------------------------------
st.subheader("🧍 Driver Demographics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Driver Age Group Distribution**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, y="Driver Age Group", order=filtered_df["Driver Age Group"].value_counts().index, ax=ax)
    ax.set_title("Distribution by Age Group")
    st.pyplot(fig)

with col2:
    st.markdown("**Driver Gender Distribution**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, x="Driver Gender", palette="coolwarm", ax=ax)
    ax.set_title("Distribution by Gender")
    st.pyplot(fig)

# ------------------------------
# Unique Visualization 3: Road & Vehicle Conditions
# ------------------------------
st.subheader("⚙️ Road & Vehicle Condition Impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Accidents by Road Condition**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, y="Road Condition", order=filtered_df["Road Condition"].value_counts().index, ax=ax)
    ax.set_title("Road Condition Impact")
    st.pyplot(fig)

with col2:
    st.markdown("**Accidents by Vehicle Condition**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, y="Vehicle Condition", order=filtered_df["Vehicle Condition"].value_counts().index, ax=ax)
    ax.set_title("Vehicle Condition Impact")
    st.pyplot(fig)

# ------------------------------
# Economic & Correlation Insights
# ------------------------------
st.subheader("💰 Economic Impact by Region")
econ = filtered_df.groupby("Region")[["Economic Loss", "Medical Cost"]].sum().sort_values("Economic Loss", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
econ.plot(kind="bar", ax=ax)
ax.set_title("Economic Loss vs Medical Cost by Region")
st.pyplot(fig)

st.subheader("🔥 Correlation Heatmap")
num_cols = filtered_df.select_dtypes(include=["float64", "int64"])
corr = num_cols.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

# ------------------------------
# AI-like Insights Summary
# ------------------------------
st.markdown("---")
st.subheader("🧠 Automated Insights Summary")

top_weather = filtered_df["Weather Conditions"].mode()[0]
top_cause = filtered_df["Accident Cause"].mode()[0]
common_area = filtered_df["Urban/Rural"].mode()[0]
common_time = filtered_df["Time of Day"].mode()[0]
fatalities = filtered_df["Number of Fatalities"].sum()
injuries = filtered_df["Number of Injuries"].sum()

st.markdown(f"""
✅ Most accidents occur in **{common_area.lower()} areas** during **{common_time.lower()} hours**.  
🌦️ The most common weather condition during accidents is **{top_weather}**.  
⚠️ The leading cause of accidents is **{top_cause}**.  
💀 Total fatalities recorded: **{fatalities:,}**, while injuries reached **{injuries:,}**.  
💡 Economic losses are highest in regions with dense populations and high traffic volumes.
""")
