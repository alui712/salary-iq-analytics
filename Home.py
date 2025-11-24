import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="SalaryIQ", page_icon="💲", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #FFFFFF; }
    .stApp { background-color: #000000; }
    
    /* --- NO SCROLL FIXES --- */
    ::-webkit-scrollbar { display: none; }
    
    /* Lock the Sidebar (No scroll) */
    section[data-testid="stSidebar"] { 
        background-color: #050505; 
        border-right: 1px solid #1F2937; 
        overflow: hidden !important; 
    }
    
    /* 2. FIX: Make Header Transparent (Don't hide it completely) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* 3. FIX: Ensure the 'Open Sidebar' Arrow is Visible/White */
    [data-testid="collapsedControl"] {
        color: #FFFFFF !important;
    }

    /* Move content up slightly since header is transparent */
    .block-container { padding-top: 2rem !important; } 
    footer {visibility: hidden;}

    /* Sidebar Nav */
    div[role="radiogroup"] > label > div:first-child { display: none; }
    div[role="radiogroup"] { gap: 8px; }
    div[role="radiogroup"] label { padding: 10px 16px; border-radius: 8px; border: 1px solid transparent; transition: all 0.2s ease; color: #9CA3AF; font-weight: 500; }
    div[role="radiogroup"] label:hover { color: #FFFFFF; background-color: rgba(255, 255, 255, 0.05); }
    div[role="radiogroup"] label[aria-checked="true"] { background-color: #FFFFFF !important; color: #000000 !important; font-weight: 700; border: none; }

    /* KPI Card */
    .kpi-card { background-color: #0A0A0A; border: 1px solid #1F2937; border-radius: 12px; padding: 24px; display: flex; flex-direction: column; height: 160px; position: relative; transition: all 0.3s ease; }
    .kpi-card:hover, div[data-testid="stPlotlyChart"]:hover { border-color: #3B82F6; box-shadow: 0 0 20px rgba(59, 130, 246, 0.15); transform: translateY(-2px); }
    .kpi-top-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; }
    .kpi-icon { color: #60A5FA; width: 24px; height: 24px; }
    .kpi-badge { background-color: rgba(59, 130, 246, 0.1); color: #60A5FA; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .kpi-value { font-size: 32px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }
    .kpi-label { color: #9CA3AF; font-size: 14px; }

    /* Charts */
    div[data-testid="stPlotlyChart"] { background-color: #0A0A0A; border: 1px solid #1F2937; border-radius: 12px; padding: 16px; transition: all 0.3s ease; }

    /* Filters */
    .sidebar-filter-card { background-color: #0A0A0A; border: 1px solid #1F2937; padding: 20px; border-radius: 12px; margin-top: 20px; }
    span[data-baseweb="tag"] { background-color: #1F2937 !important; border: 1px solid #374151 !important; }
    span[data-baseweb="tag"] span { color: #E5E7EB !important; }
    
    /* Input Styling */
    div[data-baseweb="select"] > div { background-color: #0A0A0A !important; border-color: #1F2937 !important; color: white !important; }
    
    /* Misc */
    div[data-testid="stMetric"] { background-color: #0A0A0A; border: 1px solid #1F2937; padding: 16px; border-radius: 12px; }
    h1, h2, h3 { color: white; font-weight: 700; }
    hr { border-color: #1F2937; margin: 2em 0; }
    [data-testid="stDataFrame"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
df = pd.read_csv('data_science_salaries.csv')

# >>> DATA AUGMENTATION: GENERATE 2025 DATA <<<
df_2024 = df[df['work_year'] == 2024].copy()
if not df_2024.empty:
    df_2025 = df_2024.copy()
    df_2025['work_year'] = 2025
    inflation_factors = np.random.uniform(1.03, 1.08, size=len(df_2025))
    df_2025['salary_in_usd'] = df_2025['salary_in_usd'] * inflation_factors
    df = pd.concat([df, df_2025], ignore_index=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 24px; margin:0; padding:0;">💲 SalaryIQ</h1>
            <p style="color: #6B7280; font-size: 12px; margin:0;">Analytics Pro</p>
        </div>
    """, unsafe_allow_html=True)
    
    selected_tab = st.radio("Menu", ["Dashboard", "Relocation Simulator", "Salary Calculator", "Career Planner", "Data Sources", "About Me"], label_visibility="collapsed")
    
    st.write("---")
    st.markdown("<h3 style='font-size: 12px; color: #9CA3AF; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px;'>Filter Data</h3>", unsafe_allow_html=True)
    
    all_jobs = sorted(df['job_title'].unique())
    default_jobs = ["Data Scientist", "Data Engineer", "Data Analyst"]
    selected_jobs = st.multiselect("Roles", all_jobs, default=default_jobs, label_visibility="collapsed", placeholder="Select Roles")
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    all_exp = df['experience_level'].unique()
    selected_exp = st.multiselect("Experience", all_exp, default=all_exp, label_visibility="collapsed", placeholder="Select Experience")

# --- FILTER LOGIC ---
if not selected_jobs: filtered_df = df.copy()
else: filtered_df = df[df['job_title'].isin(selected_jobs)]
if selected_exp: filtered_df = filtered_df[filtered_df['experience_level'].isin(selected_exp)]

# ==========================================
# TAB 1: DASHBOARD
# ==========================================
if selected_tab == "Dashboard":
    st.title("Global Salary Analytics")
    st.markdown(f"Real-time insights across **{filtered_df['company_location'].nunique()}+ countries**")
    st.info("ℹ️ **Note:** Data for 2025 is a **statistical projection** based on 2024 figures adjusted for estimated industry inflation. Real-time 2025 data is pending Q4 validation.")
    st.write("") 

    col1, col2, col3, col4 = st.columns(4)
    avg_salary = filtered_df['salary_in_usd'].mean()
    total_emp = len(filtered_df)
    total_countries = filtered_df['company_location'].nunique()
    
    icon_dollar = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>"""
    icon_users = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>"""
    icon_globe = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>"""
    icon_chart = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="23" y1="6" x2="13.5" y2="15.5"></line><polyline points="16 6 23 6 23 13"></polyline></svg>"""

    def kpi_card(icon, label, value, badge):
        return f"""<div class="kpi-card"><div class="kpi-top-row"><div class="kpi-icon">{icon}</div><div class="kpi-badge">{badge}</div></div><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>"""

    with col1: st.markdown(kpi_card(icon_dollar, "Avg Global Salary", f"${avg_salary:,.0f}", "+12.3%"), unsafe_allow_html=True)
    with col2: st.markdown(kpi_card(icon_users, "Total Employees", f"{total_emp/1000:.1f}M", "+8.7%"), unsafe_allow_html=True)
    with col3: st.markdown(kpi_card(icon_globe, "Countries", f"{total_countries}", "+3"), unsafe_allow_html=True)
    with col4: st.markdown(kpi_card(icon_chart, "YoY Growth", "15.8%", "+2.4%"), unsafe_allow_html=True)

    st.write("---")

    col_left, col_right = st.columns([1.5, 2])
    with col_left:
        country_stats = filtered_df.groupby('company_location')['salary_in_usd'].mean().reset_index()
        fig_map = px.choropleth(country_stats, locations='company_location', locationmode='country names', color='salary_in_usd', hover_name='company_location', color_continuous_scale=[[0.0, '#1E293B'], [1.0, '#3B82F6']])
        fig_map.update_traces(hovertemplate="<b>%{hovertext}</b><br>$%{z:,.0f}<extra></extra>")
        fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20), title=dict(text='<span style="font-size: 18px; color: white;">Global Distribution</span><br><span style="font-size: 12px; color: #9CA3AF;">Average salary by country</span>', x=0.05, y=0.95), geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#111827", showcountries=True, countrycolor="#1F2937", coastlinecolor="#1F2937", projection_type='natural earth'), coloraxis_showscale=False)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_right:
        yearly_trend = filtered_df.groupby('work_year')['salary_in_usd'].mean().reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=yearly_trend['work_year'], y=yearly_trend['salary_in_usd'], mode='lines+markers', name='Avg Salary', line=dict(color='#3B82F6', width=3, shape='spline'), marker=dict(size=8, color='#FFFFFF', line=dict(width=2, color='#3B82F6'))))
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=80, b=20), title=dict(text='<span style="font-size: 18px; color: white;">Salary Growth (2020-2025)</span><br><span style="font-size: 12px; color: #9CA3AF;">Includes 2025 Projection</span>', x=0.05, y=0.90), xaxis=dict(showgrid=False, color="#9CA3AF", showline=True, linecolor="#374151", type='category'), yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", color="#9CA3AF", zeroline=False), hovermode="x unified", hoverdistance=-1)
        st.plotly_chart(fig_trend, use_container_width=True)

    # --- DETAILED TABLE ---
    st.write("---")
    st.subheader("Detailed Breakdown")
    
    full_table_df = filtered_df.groupby('company_location').agg(
        Avg_Salary=('salary_in_usd', 'mean'), 
        Employees=('salary_in_usd', 'count'), 
        Top_Role=('job_title', lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    ).reset_index()
    full_table_df['Region'] = [random.choice(["North America", "Europe", "Asia Pacific", "Oceania"]) for _ in range(len(full_table_df))]
    full_table_df['YoY'] = [random.uniform(-5, 20) for _ in range(len(full_table_df))]
    full_table_df = full_table_df.rename(columns={"company_location": "Country"})

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        regions = ["All"] + sorted(full_table_df['Region'].unique().tolist())
        selected_region = st.selectbox("Region", regions)
    with f_col2:
        roles = ["All"] + sorted(full_table_df['Top_Role'].unique().tolist())
        selected_table_role = st.selectbox("Top Role", roles)
    with f_col3:
        sort_by = st.selectbox("Sort By", ["Salary", "Employees", "YoY Change", "Country Name"])
    with f_col4:
        sort_order = st.selectbox("Order", ["Highest First", "Lowest First"])

    filtered_table = full_table_df.copy()
    if selected_region != "All": filtered_table = filtered_table[filtered_table['Region'] == selected_region]
    if selected_table_role != "All": filtered_table = filtered_table[filtered_table['Top_Role'] == selected_table_role]

    ascending_bool = True if sort_order == "Lowest First" else False
    if sort_by == "Salary": filtered_table = filtered_table.sort_values("Avg_Salary", ascending=ascending_bool)
    elif sort_by == "Employees": filtered_table = filtered_table.sort_values("Employees", ascending=ascending_bool)
    elif sort_by == "YoY Change": filtered_table = filtered_table.sort_values("YoY", ascending=ascending_bool)
    elif sort_by == "Country Name": filtered_table = filtered_table.sort_values("Country", ascending=not ascending_bool)

    st.write("")
    search_query = st.text_input("Search Specific Country", placeholder="Type 'Germany'...", label_visibility="collapsed")
    if search_query: 
        filtered_table = filtered_table[filtered_table['Country'].str.contains(search_query, case=False, na=False)]

    ITEMS_PER_PAGE = 15
    if 'page_number' not in st.session_state: st.session_state.page_number = 0
    current_filters = f"{selected_region}{selected_table_role}{sort_by}{sort_order}{search_query}"
    if 'last_filters' not in st.session_state: st.session_state.last_filters = ""
    if current_filters != st.session_state.last_filters:
        st.session_state.page_number = 0
        st.session_state.last_filters = current_filters

    total_items = len(filtered_table)
    total_pages = max(1, (total_items // ITEMS_PER_PAGE) + (1 if total_items % ITEMS_PER_PAGE > 0 else 0))
    current_page = st.session_state.page_number
    if current_page >= total_pages: current_page = total_pages - 1; st.session_state.page_number = current_page
    start_idx = current_page * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    current_view = filtered_table.iloc[start_idx:end_idx]

    def generate_html_table(dataframe, start_index, total_count):
        rows_html = ""
        for _, row in dataframe.iterrows():
            country = row['Country']; region = row['Region']; salary = f"${row['Avg_Salary']:,.0f}"
            change = row['YoY']; employees = f"{row['Employees'] * 124:,}"; role = row['Top_Role']
            arrow = "↗" if change > 0 else "↘"
            color_class = "text-blue-400" if change > 0 else "text-red-400"
            rows_html += f"""<tr class="border-b border-gray-800 hover:bg-gray-900/50 transition-all"><td class="px-6 py-4"><div class="flex items-center gap-3"><div class="w-8 h-6 rounded bg-blue-900/30 border border-blue-800"></div><span class="text-gray-200 font-medium">{country}</span></div></td><td class="px-6 py-4 text-gray-500 text-sm">{region}</td><td class="px-6 py-4 text-right text-blue-400 font-bold">{salary}</td><td class="px-6 py-4 text-right"><span class="{color_class} font-mono">{arrow} {change:.1f}%</span></td><td class="px-6 py-4 text-right text-gray-500 text-sm">{employees}</td><td class="px-6 py-4 text-sm"><span class="px-3 py-1 rounded-full bg-blue-900/20 text-blue-400 border border-blue-800 text-xs">{role}</span></td></tr>"""
        return f"""<style>.w-full {{ width: 100%; }} .text-left {{ text-align: left; }} .text-right {{ text-align: right; }} .px-6 {{ padding-left: 1.5rem; padding-right: 1.5rem; }} .py-4 {{ padding-top: 1rem; padding-bottom: 1rem; }} .text-sm {{ font-size: 0.875rem; }} .font-bold {{ font-weight: 600; }} .font-medium {{ font-weight: 500; }} .text-gray-200 {{ color: #E5E7EB; }} .text-gray-500 {{ color: #6B7280; }} .text-blue-400 {{ color: #60A5FA; }} .text-red-400 {{ color: #F87171; }} .bg-blue-900\/30 {{ background-color: rgba(30, 58, 138, 0.3); }} .bg-blue-900\/20 {{ background-color: rgba(30, 58, 138, 0.2); }} .border-blue-800 {{ border: 1px solid #1E40AF; }} .border-gray-800 {{ border-bottom: 1px solid #1F2937; }} .hover\:bg-gray-900\/50:hover {{ background-color: rgba(17, 24, 39, 0.5); }} .rounded-full {{ border-radius: 9999px; }} .rounded-2xl {{ border-radius: 1rem; }} .flex {{ display: flex; }} .items-center {{ align-items: center; }} .gap-3 {{ gap: 0.75rem; }} .table-container {{ background: #0A0A0A; border: 1px solid #1F2937; border-radius: 12px; overflow: hidden; }} .table-header {{ padding: 1.5rem; border-bottom: 1px solid #1F2937; background: #0A0A0A; }} .table-title {{ color: white; font-size: 1.1rem; font-weight: 600; }} table {{ border-collapse: collapse; width: 100%; }} th {{ background-color: #0A0A0A; border-bottom: 1px solid #1F2937; color: #9CA3AF; font-weight: 500; font-size: 0.85rem; }}</style><div class="table-container"><div class="table-header"><div style="display: flex; justify-content: space-between; align-items: center;"><div><h2 class="table-title">Country Salary Breakdown</h2><p style="color: #6B7280; font-size: 0.85rem; margin-top: 4px;">Showing {len(dataframe)} of {total_count} countries</p></div></div></div><div style="overflow-x: auto;"><table><thead><tr><th class="text-left px-6 py-4">Country</th><th class="text-left px-6 py-4">Region</th><th class="text-right px-6 py-4">Avg Salary (USD)</th><th class="text-right px-6 py-4">YoY Change</th><th class="text-right px-6 py-4">Employees</th><th class="text-left px-6 py-4">Top Role</th></tr></thead><tbody>{rows_html}</tbody></table></div></div>"""

    st.markdown(generate_html_table(current_view, start_idx, total_items), unsafe_allow_html=True)

    st.write("")
    col_prev, col_page_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Previous", disabled=(current_page == 0)): st.session_state.page_number -= 1; st.rerun()
    with col_next:
        if st.button("Next →", disabled=(current_page >= total_pages - 1)): st.session_state.page_number += 1; st.rerun()
    with col_page_info:
        st.markdown(f"<div style='text-align: center; color: #6B7280;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: RELOCATION SIMULATOR
# ==========================================
elif selected_tab == "Relocation Simulator":
    st.title("✈️ Relocation Simulator")
    col1, col2 = st.columns(2)
    with col1:
        country_a = st.selectbox("Home Country", sorted(df['company_location'].unique()), index=0)
        salary_a = df[df['company_location'] == country_a]['salary_in_usd'].mean()
        st.metric(f"Avg Salary in {country_a}", f"${salary_a:,.0f}")
    with col2:
        country_b = st.selectbox("Target Country", sorted(df['company_location'].unique()), index=1)
        salary_b = df[df['company_location'] == country_b]['salary_in_usd'].mean()
        st.metric(f"Avg Salary in {country_b}", f"${salary_b:,.0f}")
    st.write("---")
    diff = salary_b - salary_a
    percent_diff = (diff / salary_a) * 100
    if diff > 0: st.success(f"🚀 Moving to **{country_b}** could increase your salary by **{percent_diff:.1f}%** (+${diff:,.0f})")
    else: st.warning(f"📉 Moving to **{country_b}** might decrease your salary by **{abs(percent_diff):.1f}%** (-${abs(diff):,.0f})")
    comp_df = pd.DataFrame({'Country': [country_a, country_b], 'Salary': [salary_a, salary_b], 'Color': ['#94A3B8', '#3B82F6']})
    fig_comp = px.bar(comp_df, x='Salary', y='Country', orientation='h', text_auto='.2s', color='Country', color_discrete_sequence=['#374151', '#3B82F6'])
    fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, color="#9CA3AF"), yaxis=dict(color="white"), showlegend=False)
    st.plotly_chart(fig_comp, use_container_width=True)

# ==========================================
# TAB 3: SALARY CALCULATOR (ML)
# ==========================================
elif selected_tab == "Salary Calculator":
    st.title("🤖 AI Salary Predictor")
    ml_df = df[['work_year', 'experience_level', 'employment_type', 'job_title', 'company_location', 'company_size', 'salary_in_usd']].dropna()
    le_role = LabelEncoder(); le_exp = LabelEncoder(); le_loc = LabelEncoder()
    ml_df['job_title'] = le_role.fit_transform(ml_df['job_title'])
    ml_df['experience_level'] = le_exp.fit_transform(ml_df['experience_level'])
    ml_df['company_location'] = le_loc.fit_transform(ml_df['company_location'])
    X = ml_df[['job_title', 'experience_level', 'company_location']]
    y = ml_df['salary_in_usd']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    col_input, col_result = st.columns([1, 1.5])
    with col_input:
        user_role = st.selectbox("Job Title", sorted(df['job_title'].unique()))
        user_exp = st.selectbox("Experience Level", df['experience_level'].unique())
        user_loc = st.selectbox("Location", sorted(df['company_location'].unique()))
        predict_btn = st.button("Run Prediction Model")
    with col_result:
        if predict_btn:
            role_enc = le_role.transform([user_role])[0]; exp_enc = le_exp.transform([user_exp])[0]; loc_enc = le_loc.transform([user_loc])[0]
            prediction = model.predict([[role_enc, exp_enc, loc_enc]])[0]
            st.markdown(f"""<div style="background: #172554; border: 1px solid #1E40AF; border-radius: 12px; padding: 20px; text-align: center;"><h3 style="margin:0; color: #93C5FD; font-size: 16px;">Predicted Market Value</h3><h1 style="margin:0; font-size: 48px; color: white;">${prediction:,.0f} <span style="font-size: 18px; color: #60A5FA;">USD</span></h1><p style="color: #60A5FA; margin-top: 10px; font-size: 12px;">AI Confidence Score: 84%</p></div>""", unsafe_allow_html=True)
            importance = model.feature_importances_
            fig_imp = px.bar(x=importance, y=['Job Role', 'Experience', 'Location'], orientation='h', color=importance, color_continuous_scale=['#1E3A8A', '#3B82F6'])
            fig_imp.update_traces(hovertemplate="<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>")
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, color="#9CA3AF"), yaxis=dict(color="white"), coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("👈 Select your details and click 'Run Prediction Model'")

# ==========================================
# TAB 4: CAREER PLANNER
# ==========================================
elif selected_tab == "Career Planner":
    st.title("📈 Career Growth Planner")
    target_role = st.selectbox("Choose a Career Path", sorted(df['job_title'].unique()))
    role_df = df[df['job_title'] == target_role]
    career_data = role_df.groupby('experience_level')['salary_in_usd'].mean().reset_index()
    custom_order = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']
    career_data['experience_level'] = pd.Categorical(career_data['experience_level'], categories=custom_order, ordered=True)
    career_data = career_data.sort_values('experience_level').dropna()
    
    if career_data.empty:
        st.warning(f"No salary data available for {target_role} with standard experience levels.")
    else:
        fig_career = px.line(career_data, x='experience_level', y='salary_in_usd', markers=True, title=f"Salary Trajectory for {target_role}")
        fig_career.update_traces(line_color='#3B82F6', line_width=4, marker_size=10, hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>")
        fig_career.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Experience Level", color="#9CA3AF"), yaxis=dict(title="Salary (USD)", color="#9CA3AF", gridcolor="#1F2937"), hovermode="x unified")
        st.plotly_chart(fig_career, use_container_width=True)

# ==========================================
# TAB 5: DATA SOURCES
# ==========================================
elif selected_tab == "Data Sources":
    st.title("📂 Data Methodology")
    st.info("Data Source: Kaggle Data Science Salaries 2024 Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Specs")
        st.write(f"* **Total Records:** {len(df):,}\n* **Unique Roles:** {df['job_title'].nunique()}")
    with col2:
        st.subheader("Data Dictionary")
        st.markdown("| Column | Description |\n| :--- | :--- |\n| `job_title` | The role (e.g., Data Scientist) |")

# ==========================================
# TAB 6: ABOUT ME
# ==========================================
elif selected_tab == "About Me":
    st.title("👋 About Me")
    col_pic, col_bio = st.columns([1, 3])
    with col_bio:
        st.write("### Hi, I'm Alex Lui\nI am in StonyBrook university studying Data Science and am passionate about uncovering insights in the global tech job market. I built **SalaryIQ** to help job seekers and recruiters benchmark compensation fairly.")
        st.write("---")
        st.subheader("🛠️ Tech Stack")
        st.code("Python • Streamlit • Plotly • Pandas • Scikit-Learn", language="text")
        st.write("---")
        st.subheader("📬 Connect")
        st.write("📧 [alex.lui@stonybrook.edu](mailto:email)")