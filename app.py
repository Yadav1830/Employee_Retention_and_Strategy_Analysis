import streamlit as st
import pandas as pd
import numpy as np
#import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

# -----------------------------------------------------
# DATA LOADING
# -----------------------------------------------------

st.sidebar.title("Dataset")

try:
    df = pd.read_csv("cleaned_hr_data.csv")
    st.sidebar.success("Default dataset loaded")
except:
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload dataset to continue")
        st.stop()

# -----------------------------------------------------
# DATA PREPROCESSING
# -----------------------------------------------------

if "Attrition" in df.columns:
    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})

# Age groups
bins = [18,25,35,45,55,60]
labels = ['18-25','26-35','36-45','46-55','56+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------

st.sidebar.title("Filters")

department = st.sidebar.multiselect(
    "Department",
    df["Department"].unique(),
    default=df["Department"].unique()
)

overtime = st.sidebar.multiselect(
    "Overtime",
    df["OverTime"].unique(),
    default=df["OverTime"].unique()
)

filtered_df = df[
    (df["Department"].isin(department)) &
    (df["OverTime"].isin(overtime))
]

# -----------------------------------------------------
# PAGE NAVIGATION
# -----------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Summary & KPIs","Visualizations","Prediction"]
)

# =====================================================
# PAGE 1 : SUMMARY + KPI
# =====================================================

if page == "Summary & KPIs":

    st.title("Employee Attrition Dashboard")

    total_emp = len(filtered_df)
    left_emp = filtered_df["Attrition"].sum()
    attrition_rate = round(left_emp/total_emp*100,2)

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Employees", total_emp)
    col2.metric("Employees Left", left_emp)
    col3.metric("Attrition Rate %", attrition_rate)

    st.subheader("Summary Statistics")

    st.dataframe(filtered_df.describe())

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head())

# =====================================================
# PAGE 2 : VISUALIZATIONS
# =====================================================

elif page == "Visualizations":

    st.title("Attrition Analysis Visualizations")

    # Q1 Attrition percentage
    st.subheader("Q1 Employee Attrition Percentage")

    attr = filtered_df["Attrition"].value_counts().reset_index()
    attr.columns=["Status","Count"]
    attr["Status"]=attr["Status"].map({0:"Stayed",1:"Left"})

    fig = px.bar(attr,x="Status",y="Count",
                 color="Status",
                 title="Attrition Percentage")

    st.plotly_chart(fig,use_container_width=True)

    # Q2 Department attrition
    st.subheader("Q2 Department with Highest Attrition")

    dept = filtered_df.groupby("Department")["Attrition"].mean().reset_index()

    fig = px.bar(dept,x="Department",y="Attrition",
                 color="Department")

    st.plotly_chart(fig,use_container_width=True)

    # Q3 Salary vs attrition
    st.subheader("Q3 Salary Impact on Attrition")

    fig = px.box(filtered_df,
                 x="Attrition",
                 y="MonthlyIncome",
                 color="Attrition")

    st.plotly_chart(fig,use_container_width=True)

    # Q4 Overtime effect
    st.subheader("Q4 Overtime vs Attrition")

    fig = px.histogram(filtered_df,
                       x="OverTime",
                       color="Attrition",
                       barmode="group")

    st.plotly_chart(fig,use_container_width=True)

    # Q5 Age group attrition
    st.subheader("Q5 Attrition by Age Group")

    fig = px.histogram(filtered_df,
                       x="AgeGroup",
                       color="Attrition")

    st.plotly_chart(fig,use_container_width=True)

    # Q6 Distance from home
    st.subheader("Q6 Distance from Home vs Attrition")

    fig = px.box(filtered_df,
                 x="Attrition",
                 y="DistanceFromHome",
                 color="Attrition")

    st.plotly_chart(fig,use_container_width=True)

    # Q7 Job Satisfaction
    st.subheader("Q7 Job Satisfaction vs Attrition")

    fig = px.histogram(filtered_df,
                       x="JobSatisfaction",
                       color="Attrition")

    st.plotly_chart(fig,use_container_width=True)

    # Q8 Work Life Balance
    st.subheader("Q8 Work Life Balance vs Attrition")

    fig = px.histogram(filtered_df,
                       x="WorkLifeBalance",
                       color="Attrition")

    st.plotly_chart(fig,use_container_width=True)

    # Q9 Correlation
    st.subheader("Q9 Correlation Heatmap")

    numeric = filtered_df.select_dtypes(include=np.number)

    fig = px.imshow(numeric.corr(),
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix")

    st.plotly_chart(fig,use_container_width=True)

    # Q10 Key Factors
    st.subheader("Q10 Key Factors Affecting Attrition")

    factors = filtered_df.groupby("Attrition")[[
        "MonthlyIncome",
        "JobSatisfaction",
        "WorkLifeBalance",
        "DistanceFromHome",
        "Age"
    ]].mean()

    st.dataframe(factors)

# =====================================================
# PAGE 3 : PREDICTION
# =====================================================

elif page == "Prediction":

    st.title("Employee Attrition Prediction")

    features = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "JobSatisfaction",
        "WorkLifeBalance"
    ]

    X = filtered_df[features]
    y = filtered_df["Attrition"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    st.metric("Model Accuracy",round(acc*100,2))

    st.subheader("Predict Attrition for New Employee")

    age = st.slider("Age",18,60,30)
    salary = st.number_input("Monthly Income",1000,20000,5000)
    distance = st.slider("Distance From Home",1,30,10)
    job_sat = st.slider("Job Satisfaction",1,4,2)
    work_life = st.slider("Work Life Balance",1,4,2)

    input_data = pd.DataFrame({
        "Age":[age],
        "MonthlyIncome":[salary],
        "DistanceFromHome":[distance],
        "JobSatisfaction":[job_sat],
        "WorkLifeBalance":[work_life]
    })

    if st.button("Predict"):
        result = model.predict(input_data)

        if result[0]==1:
            st.error("Employee likely to leave")
        else:
            st.success("Employee likely to stay")