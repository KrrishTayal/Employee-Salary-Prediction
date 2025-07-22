import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json

@st.cache_resource
def load_model():
    return joblib.load('models/salary_predictor.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/salary_data.csv')

@st.cache_data
def load_column_order():
    with open('models/column_order.json', 'r') as f:
        return json.load(f)

def preprocess_input(input_dict, column_order):
    data = {col: 0 for col in column_order}
    for feat in ['age', 'experience', 'education', 'job_title', 'location', 'company_size']:
        data[feat] = input_dict[feat]
    for skill in input_dict['skills']:
        skill_col = f'skill_{skill.lower().replace(" ", "_")}'
        if skill_col in column_order:
            data[skill_col] = 1
    input_df = pd.DataFrame([data])[column_order]
    for col in column_order:
        if col.startswith('skill_'):
            input_df[col] = input_df[col].astype(int)
    return input_df

def main():
    df = load_data()
    model = load_model()
    column_order = load_column_order()

    st.set_page_config(page_title="AI Salary Predictor", layout="wide")

    st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .result-box {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0px;
        background-color: #f0f8ff;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title(" AI-Powered Salary Predictor")
    st.markdown("Predict your salary based on job market trends")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Salary Prediction", "Career Growth Projection", "Top Paying Roles"], key="main_nav_radio")

    if app_mode == "Salary Prediction":
        render_salary_prediction(df, model, column_order)
    elif app_mode == "Career Growth Projection":
        render_career_growth(df, model, column_order)
    elif app_mode == "Top Paying Roles":
        render_top_paying_roles(df)

def render_salary_prediction(df, model, column_order):
    st.header(" AI-Powered Salary Estimation")

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.selectbox("Job Title", sorted(df['job_title'].unique()))
        location = st.selectbox("Location", sorted(df['location'].unique()))
        company_size = st.selectbox("Company Size", sorted(df['company_size'].unique()))
    with col2:
        experience = st.slider("Years of Experience", 0, 40, 5)
        education = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
        age = st.slider("Age", 22, 60, 30)

    st.subheader("Skills")
    top_skills = ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript', 'Financial Modeling', 'Project Management', 'Leadership']
    skills = []
    cols_per_row = 4

    for row_start in range(0, len(top_skills), cols_per_row):
        row_skills = top_skills[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_skills))
        for col, skill in zip(cols, row_skills):
            with col:
                if st.checkbox(skill, key=f"skill_{skill}"):
                    skills.append(skill)


    if st.button("Predict Salary"):
        input_data = {
            'age': age,
            'experience': experience,
            'education': education,
            'job_title': job_title,
            'location': location,
            'company_size': company_size,
            'skills': skills
        }

        try:
            input_df = preprocess_input(input_data, column_order)
            prediction = model.predict(input_df)[0]

            st.markdown(f"""
            <div class="result-box">
                <p class="big-font">Predicted Salary: ${prediction:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            confidence = max(0.85, min(0.98, 0.9 + (experience / 100)))
            lower_bound = int(prediction * (1 - (1 - confidence)/2))
            upper_bound = int(prediction * (1 + (1 - confidence)/2))

            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Salary Prediction with Confidence"},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [lower_bound * 0.9, upper_bound * 1.1]},
                    'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': prediction},
                    'steps': [
                        {'range': [lower_bound * 0.9, lower_bound], 'color': "lightgray"},
                        {'range': [lower_bound, upper_bound], 'color': "lightgreen"},
                        {'range': [upper_bound, upper_bound * 1.1], 'color': "lightgray"}
                    ],
                    'bar': {'color': "darkblue"}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            peers = df[
                (df['job_title'] == job_title) &
                (df['experience'].between(experience-2, experience+2)) &
                (df['education'] == education)
            ]

            if not peers.empty:
                avg_salary = peers['salary'].mean()
                percentile = np.mean(prediction > peers['salary']) * 100

                st.subheader("Comparison with Peers")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average for Similar Profiles", f"${avg_salary:,.0f}")
                    st.metric("Your Percentile", f"{percentile:.1f}%")
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=peers['salary'], name='Peers', boxpoints='all', jitter=0.3, pointpos=-1.8))
                    fig.add_trace(go.Scatter(x=['Peers'], y=[prediction], mode='markers', marker=dict(color='red', size=12), name='Your Prediction'))
                    fig.update_layout(title='Salary Distribution Comparison', yaxis_title='Salary')
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def render_career_growth(df, model, column_order):
    st.header(" Career Growth & Salary Projection")

    col1, col2 = st.columns(2)
    with col1:
        current_job = st.selectbox("Current Job Title", sorted(df['job_title'].unique()), key="current_job")
        current_exp = st.slider("Current Experience (years)", 0, 40, 5, key="current_exp")
        current_edu = st.selectbox("Current Education Level", ['High School', 'Bachelor', 'Master', 'PhD'], key="current_edu")
    with col2:
        future_job = st.selectbox("Future Job Title (promotion target)", sorted(df['job_title'].unique()), key="future_job")
        years_to_project = st.slider("Years to Project", 1, 15, 5, key="years_proj")
        future_edu = st.selectbox("Future Education Goal", ['Same as Current', 'Bachelor', 'Master', 'PhD'], key="future_edu")

    st.subheader("Future Skills to Acquire")
    top_skills = ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript', 'Financial Modeling', 'Project Management', 'Leadership']
    skills = []
    cols_per_row = 4

    for row_start in range(0, len(top_skills), cols_per_row):
        row_skills = top_skills[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_skills))
        for col, skill in zip(cols, row_skills):
            with col:
                if st.checkbox(skill, key=f"skill_{skill}"):
                    skills.append(skill)


    if st.button("Generate Projection", key="proj_button"):
        years = list(range(current_exp, current_exp + years_to_project + 1))
        salaries, skill_progress = [], []

        for i, year in enumerate(years):
            progress = i / len(years)
            current_skills = []
            if progress > 0.5:
                num_skills_to_add = int(len(future_skills) * min(1, (progress - 0.5) * 2))
                current_skills = future_skills[:num_skills_to_add]

            input_data = {
                'age': 30 + (year - current_exp),
                'experience': year,
                'education': current_edu if future_edu == 'Same as Current' else future_edu,
                'job_title': current_job if progress < 0.5 else future_job,
                'location': 'Remote',
                'company_size': 'Medium (200-1000)',
                'skills': current_skills
            }
            input_df = preprocess_input(input_data, column_order)
            salaries.append(model.predict(input_df)[0])
            skill_progress.append(", ".join(current_skills) if current_skills else "None")

        projection_df = pd.DataFrame({
            'Year': [2023 + (year - current_exp) for year in years],
            'Experience': years,
            'Salary': salaries,
            'New Skills': skill_progress
        })

        fig1 = px.line(projection_df, x='Year', y='Salary', title="Salary Projection Over Time", markers=True, hover_data=['New Skills'])
        fig1.update_traces(line_color='#4CAF50', line_width=3)
        fig1.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Skill Acquisition Timeline")
        fig2 = px.bar(projection_df, x='Year', y=[1]*len(projection_df), color='New Skills', title="When New Skills Will Be Added", labels={'New Skills': 'Skills Added'})
        fig2.update_layout(showlegend=True, yaxis_visible=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Key Milestones")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Starting Salary", f"${salaries[0]:,.0f}")
            st.metric("1 Year Growth", f"${salaries[1] - salaries[0]:,.0f}", delta=f"{((salaries[1]/salaries[0])-1)*100:.1f}%")
        with col2:
            st.metric(f"After {years_to_project} Years", f"${salaries[-1]:,.0f}")
            st.metric("Total Growth", f"${salaries[-1] - salaries[0]:,.0f}", delta=f"{((salaries[-1]/salaries[0])-1)*100:.1f}%")

def render_top_paying_roles(df):
    st.header(" Top Paying Roles")

    avg_salary_by_job = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
    top_jobs = avg_salary_by_job.head(10)
    fig = px.bar(top_jobs, x=top_jobs.values, y=top_jobs.index, orientation='h', title="Top 10 Highest Paying Job Titles", labels={'x': 'Average Salary', 'y': 'Job Title'}, color=top_jobs.values, color_continuous_scale='greens')
    fig.update_layout(xaxis_tickprefix='$')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Salary by Company Size")
    fig = px.box(df, x='company_size', y='salary', color='company_size', title="Salary Distribution by Company Size")
    fig.update_layout(yaxis_tickprefix='$')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
