import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Employee Retention Strategy Analysis", layout="wide")

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
    st.markdown("""
<div style='background: linear-gradient(90deg,#14B8A6,#22C55E);
            padding:11px;
            border-radius:10px;
            text-align:left;
            color:black;
            font-size:30px;
            font-weight:900;
            background-color:#245B4E;
            margin-bottom:20px;'>

<b>Employee Retention Strategy Analysis Dashboard</b>

</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div style='background: linear-gradient(90deg,#14B8A6,#22C55E);;
            padding:10px;
            border-radius:8px;
            text-align:left;
            color:black;
            font-size:22px;
            font-weight:600'>
Employee Summary & Key Metrics
</div>
""", unsafe_allow_html=True)

    

    total_emp = len(filtered_df)
    left_emp = filtered_df["Attrition"].sum()
    attrition_rate = round(left_emp/total_emp*100,2)
    
    st.markdown("""
<style>

/* KPI CARD */
[data-testid="stMetric"] {
background-color:#0F172A;
justify-content:center;              
border-radius: 10px;
padding: 15px;
border: 1px solid #14B8A6;
text-align: center !important;
box-shadow: 0px 4px 12px rgba(0,0,0,0.6);
}

/* KPI LABEL */
[data-testid="stMetricLabel"] {
color:#14B8A6;
font-weight: 600;
font-size: 16px;
}

/* KPI VALUE */
[data-testid="stMetricValue"] {
color:#E8F5E9;
font-size: 34px;
font-weight: bold;
}

</style>
""", unsafe_allow_html=True)
    employees_stayed = total_emp - left_emp
    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Employees", total_emp)
    col2.metric("Employees Left", left_emp)
    col3.metric("Employees Stayed", employees_stayed)
    col4.metric("Attrition Rate", f"{attrition_rate}%")
    

    st.subheader("Summary Statistics")

    st.dataframe(filtered_df.describe())


    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head())

# =====================================================
# PAGE 2 : VISUALIZATIONS
# =====================================================

elif page == "Visualizations":

    #st.title("Attrition Analysis Visualizations")

    # Q1 Attrition percentage
    
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q1: What Percentage of Employees Leave the Company?
</h3>
""",
unsafe_allow_html=True
)
    
    # Count employees
    attr = filtered_df["Attrition"].value_counts().reset_index()
    attr.columns = ["Attrition","Count"]

    # Map values for readability
    attr["Attrition"] = attr["Attrition"].map({0:"Stayed",1:"Left"})

    # Donut Chart
    fig = px.pie(
       attr,
       names="Attrition",
       values="Count",
       hole=0.5,
       color="Attrition",
       title="Employee Attrition Distribution",
       color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Insight:** This chart shows the proportion of employees who stayed versus those who left the company. " \
    "A higher percentage of employees leaving indicates potential retention challenges.")
    
    
    # Q2 Department attrition
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q2: Which Department Has the Highest Attrition?
</h3>
""",
unsafe_allow_html=True
)

    # Calculate attrition percentage by department
    dept = filtered_df.groupby("Department")["Attrition"].mean().reset_index()
    dept["Attrition"] = dept["Attrition"] * 100

    # Create bar chart
    fig = px.bar(
     dept,
     x="Department",
     y="Attrition",
     color="Department",
     text=dept["Attrition"].round(1),
     title="Attrition Rate by Department (%)",
     color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(yaxis_title="Attrition Rate (%)")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: This chart highlights which department experiences the highest employee attrition, " \
    "helping HR identify departments where retention strategies are most needed.")
    
    # Q3 Salary vs attrition
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q3: Does Salary Affect Attrition?
</h3>
""",
unsafe_allow_html=True
)

    # Calculate average salary by attrition
    salary = filtered_df.groupby("Attrition")["MonthlyIncome"].mean().reset_index()

    salary["Attrition"] = salary["Attrition"].map({0:"Stayed",1:"Left"})

    fig = px.bar(
     salary,
     x="Attrition",
     y="MonthlyIncome",
     color="Attrition",
     text=salary["MonthlyIncome"].round(0),
     title="Average Salary of Employees Who Stayed vs Left",
     color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(
     yaxis_title="Average Monthly Income",
     xaxis_title="Employee Status"
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Employees who left the company tend to have lower average salaries compared to those who stayed, " \
    "indicating salary may influence attrition.")

    # Q4 Overtime effect
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q4: Does Overtime Increase Attrition?
</h3>
""",
unsafe_allow_html=True
)

    # Calculate attrition rate by overtime
    overtime_attr = filtered_df.groupby("OverTime")["Attrition"].mean().reset_index()
    overtime_attr["Attrition"] = overtime_attr["Attrition"] * 100

    fig = px.bar(
     overtime_attr,
     x="OverTime",
     y="Attrition",
     color="OverTime",
     text=overtime_attr["Attrition"].round(1),
     title="Attrition Rate for Employees Working Overtime (%)",
     color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(
     xaxis_title="Overtime",
     yaxis_title="Attrition Rate (%)"
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.success("Insight: Employees who work overtime tend to have a higher attrition rate compared to those who do not work overtime, " \
    "indicating workload may influence employee turnover.")

    # Q5 Age group attrition
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q5: Which Age Group Leaves the Most?
</h3>
""",
unsafe_allow_html=True
)

   # Calculate attrition rate by age group
    age_attr = filtered_df.groupby("AgeGroup")["Attrition"].mean().reset_index()
    age_attr["Attrition"] = age_attr["Attrition"] * 100

    # Fix age group order
    age_attr["AgeGroup"] = pd.Categorical(
    age_attr["AgeGroup"],
    ["18-25","26-35","36-45","46-55","56+"]
)

    fig = px.bar(
     age_attr,
     x="AgeGroup",
     y="Attrition",
     color="AgeGroup",
     text=age_attr["Attrition"].round(1),
     title="Attrition Rate by Age Group (%)",
     color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(
     xaxis_title="Age Group",
     yaxis_title="Attrition Rate (%)"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success("Insight: The chart shows which age group experiences the highest attrition. " \
    "Younger employees often show higher attrition as they tend to explore better career opportunities.")

    # Q6 Distance from home
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q6: Does Distance From Home Influence Attrition?
</h3>
""",
unsafe_allow_html=True
)
    # Calculate average distance
    distance = filtered_df.groupby("Attrition")["DistanceFromHome"].mean().reset_index()

    distance["Attrition"] = distance["Attrition"].map({0:"Stayed",1:"Left"})

    fig = px.bar(
     distance,
     x="Attrition",
     y="DistanceFromHome",
     color="Attrition",
     text=distance["DistanceFromHome"].round(1),
     title="Average Distance from Home: Stayed vs Left",
     color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(
     xaxis_title="Employee Status",
     yaxis_title="Average Distance from Home"
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Employees who live farther from the workplace tend to show slightly higher attrition, " \
    "suggesting commute distance may influence employee retention.")

    # Q7 Job Satisfaction
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q7: Does Job Satisfaction Affect Employee Attrition?
</h3>
""",
unsafe_allow_html=True
)

    # Calculate attrition rate
    job_sat = filtered_df.groupby("JobSatisfaction")["Attrition"].mean().reset_index()
    job_sat["Attrition"] = job_sat["Attrition"] * 100

    # Add labels
    job_sat["JobSatisfaction"] = job_sat["JobSatisfaction"].map({
    1:"Low",
    2:"Medium",
    3:"High",
    4:"Very High"
    })

    fig = px.bar(
     job_sat,
     x="JobSatisfaction",
     y="Attrition",
     color="JobSatisfaction",
     text=job_sat["Attrition"].round(1),
     title="Attrition Rate by Job Satisfaction Level (%)",
     color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(
     xaxis_title="Job Satisfaction Level",
     yaxis_title="Attrition Rate (%)"
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Employees with lower job satisfaction levels tend to have higher attrition rates, " \
    "indicating that improving employee satisfaction may help reduce turnover.")

    # Q8 Work Life Balance
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q8: Does Work Life Balance Affect Employee Retention?
</h3>
""",
unsafe_allow_html=True
)

    wlb = filtered_df.groupby("WorkLifeBalance")["Attrition"].mean().reset_index()
    wlb["Attrition"] = wlb["Attrition"] * 100

    # labels for better understanding
    wlb["WorkLifeBalance"] = wlb["WorkLifeBalance"].map({
    1:"Poor",
    2:"Average",
    3:"Good",
    4:"Excellent"
    })

    fig = px.bar(
    wlb,
    x="WorkLifeBalance",
    y="Attrition",
    color="WorkLifeBalance",
    text=wlb["Attrition"].round(1),
    title="Attrition Rate by Work Life Balance (%)",
    color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(
    xaxis_title="Work Life Balance Level",
    yaxis_title="Attrition Rate (%)"
   )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Employees with poor work life balance show higher attrition, " \
    "indicating that improving work life balance may help reduce employee turnover.")

    # Q9 Correlation
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q9: Correlation Between Factors Affecting Attrition
</h3>
""",
unsafe_allow_html=True
)

    # Select important numeric columns
    corr = filtered_df[[
    "Attrition",
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "JobSatisfaction",
    "WorkLifeBalance",
    "EnvironmentSatisfaction"
    ]].corr()

    # Get correlation with Attrition only
    attrition_corr = corr["Attrition"].drop("Attrition").sort_values()

    fig = px.bar(
    attrition_corr,
    x=attrition_corr.values,
    y=attrition_corr.index,
    orientation='h',
    color=attrition_corr.values,
    title="Factors Correlated with Employee Attrition",
    color_discrete_sequence=["#14B8A6","#22C55E","#4ADE80"]
    )

    fig.update_layout(
    xaxis_title="Correlation Strength",
    yaxis_title="Factors"
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Factors such as job satisfaction, work-life balance, "
    "and environment satisfaction show stronger relationships with employee attrition compared to demographic variables.")

    # Q10 Key Factors
    st.markdown(
"""
<h3 style='color:#14B8A6'>
Q10: Key Factore affecting Attrition
</h3>
""",
unsafe_allow_html=True
)

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
    filtered_df["OverTime"] = filtered_df["OverTime"].map({"Yes":1,"No":0})

    features = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "JobSatisfaction",
        "WorkLifeBalance",
        "OverTime"
    ]

    X = filtered_df[features]
    y = filtered_df["Attrition"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train,y_train)
    print("Model Accuracy:", model.score(X_test, y_test))

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    st.metric("Model Accuracy",round(acc*100,2))

    st.subheader("Predict Attrition for New Employee")

    age = st.slider("Age",18,60,30)
    salary = st.number_input("Monthly Income",1000,20000,5000)
    distance = st.slider("Distance From Home",1,30,10)
    job_sat = st.slider("Job Satisfaction",1,4,2)
    work_life = st.slider("Work Life Balance",1,4,2)
    overtime = st.selectbox("OverTime", ["No","Yes"])

    input_data = pd.DataFrame({
        "Age":[age],
        "MonthlyIncome":[salary],
        "DistanceFromHome":[distance],
        "JobSatisfaction":[job_sat],
        "WorkLifeBalance":[work_life],
        "OverTime":[1 if overtime=="Yes" else 0]
    })
    st.write(input_data)
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.write(f"Attrition Risk Score: {round(probability*100,2)} %")

        if probability > 0.40:
         st.error("Employee likely to leave")
        else:
         st.success("Employee likely to stay")
