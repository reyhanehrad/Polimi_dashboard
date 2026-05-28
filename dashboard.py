"""
==========================================================================
GROUP 14 — Software Prototype Dashboard
Maintenance Risk Predictor + Portfolio Analytics + Building Deep Dive
==========================================================================
Run locally:  streamlit run dashboard.py
Run in Colab: see RUN_DASHBOARD_GUIDE.md
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb

# ==========================================================================
# PAGE CONFIG
# ==========================================================================
st.set_page_config(
    page_title="POLIMI Maintenance Dashboard — Group 14",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1F4E79 0%, #2E75B6 100%);
        padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;
    }
    .metric-card {
        background: #F8F9FA !important; padding: 15px; border-radius: 8px;
        border-left: 4px solid #1F4E79; color: #222 !important;
    }
    .metric-card h4, .metric-card p {
        color: #222 !important;
    }
    .risk-high   { background: #FFEBEE !important; border-left-color: #C62828; color: #222 !important; }
    .risk-medium { background: #FFF8E1 !important; border-left-color: #F57F17; color: #222 !important; }
    .risk-low    { background: #E8F5E9 !important; border-left-color: #2E7D32; color: #222 !important; }
    .risk-high h4, .risk-high p,
    .risk-medium h4, .risk-medium p,
    .risk-low h4, .risk-low p { color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================================
# DATA + MODELS
# ==========================================================================
@st.cache_data
def load_data():
    return pd.read_csv('milano_cleaned.csv', low_memory=False)

@st.cache_resource
def load_models():
    clf = xgb.XGBClassifier(); clf.load_model('model_xgb_classification.json')
    reg = xgb.XGBRegressor();  reg.load_model('model_xgb_regression.json')
    return clf, reg

@st.cache_data
def prepare_features(df):
    num_features = ['SLA_hours', 'urgency_level',
                    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
                    'hour_sin', 'hour_cos']
    df = df.copy()
    df['Management'] = df['Management'].fillna('UNK')
    mgmt_dummies = pd.get_dummies(df['Management'], prefix='mgmt', dtype=int)
    cat_dummies  = pd.get_dummies(df['macro_category'], prefix='cat', dtype=int)
    building_codes = df['ID_Building'].astype('category').cat.codes.values
    X = pd.concat([
        df[num_features].astype(float),
        mgmt_dummies, cat_dummies,
        pd.Series(building_codes, name='building_code', index=df.index)
    ], axis=1)
    return X, list(X.columns), mgmt_dummies.columns.tolist(), cat_dummies.columns.tolist()

try:
    df = load_data()
    clf_model, reg_model = load_models()
    X_full, feature_cols, mgmt_cols, cat_cols = prepare_features(df)
    shift_val = abs(df['delay_hours'].min()) + 1
    df['pred_proba']       = clf_model.predict_proba(X_full)[:, 1]
    df['pred_delay_log']   = reg_model.predict(X_full)
    df['pred_delay_hours'] = (np.expm1(df['pred_delay_log']) - shift_val).clip(lower=0)
    urgency_mult = {0: 1.0, 1: 1.5, 2: 2.0}
    df['urgency_mult']     = df['urgency_level'].map(urgency_mult)
    df['RPS']              = df['pred_proba'] * df['pred_delay_hours'] * df['urgency_mult']
except Exception as e:
    st.error(f"⚠️ Error loading data/models: {e}")
    st.info("Make sure these files are present: milano_cleaned.csv, model_xgb_classification.json, model_xgb_regression.json")
    st.stop()

# ==========================================================================
# SIDEBAR
# ==========================================================================
st.sidebar.markdown("### 🏛️ POLIMI Maintenance")
st.sidebar.markdown("**Group 14** · A.Y. 2024-2025")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📋 Navigation",
    ["🎯 Risk Predictor", "📊 Portfolio Analytics", "🏢 Building Deep Dive"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Dataset summary**
- 🎫 {len(df):,} tickets
- 🏢 {df['ID_Building'].nunique()} buildings
- 📅 Year 2023
- ⚠️ {df['is_delayed'].mean()*100:.1f}% delayed
""")
st.sidebar.markdown("---")
st.sidebar.caption("Powered by **XGBoost** + **SHAP**")

# ==========================================================================
# PAGE 1 — RISK PREDICTOR
# ==========================================================================
if page == "🎯 Risk Predictor":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">🎯 Maintenance Ticket Risk Predictor</h1>
        <p style="margin:5px 0 0 0;">Enter ticket details and get an instant prediction of SLA breach risk</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.subheader("📝 Ticket Information")
        urgency = st.selectbox("Urgency level", ["No emergency", "Urgency", "Emergency"])
        sla_options = {'45 min (0.75h)': 0.75, '60 min (1h)': 1.0, '120 min (2h)': 2.0,
                       '3h': 3.0, '5h': 5.0, '12h': 12.0, '24h': 24.0, '10 days (240h)': 240.0}
        sla_label = st.selectbox("SLA (Expiration Time)", list(sla_options.keys()), index=5)
        sla_hours = sla_options[sla_label]
        category = st.selectbox("Maintenance category",
                                sorted(df['macro_category'].dropna().unique()))
        category_full = {'ar': 'Architecture', 'av': 'Audio-Video', 'ce': 'Summer Climate',
                         'ci': 'Winter Climate', 'el': 'Electrical', 'ie': 'Elevators',
                         'is': 'Water-Sanitary', 'me': 'Edile (Masonry)', 'ms': 'Window Frames',
                         'mv': 'Green areas', 'pu': 'Cleaning'}
        st.caption(f"_{category_full.get(category, 'General')}_")
        management = st.selectbox("Management contractor",
                                  sorted(df['Management'].fillna('UNK').unique()))
        building = st.selectbox("Building ID", sorted(df['ID_Building'].unique()))

        col_dt1, col_dt2 = st.columns(2)
        with col_dt1: opening_hour = st.slider("Opening hour", 0, 23, 10)
        with col_dt2: opening_month = st.slider("Opening month", 1, 12, 6)
        opening_dow = st.select_slider("Day of week", options=list(range(7)), value=2,
            format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])

        predict_btn = st.button("🔮 Predict Risk", type="primary", use_container_width=True)

    with col_output:
        if predict_btn:
            urgency_level = {"No emergency": 0, "Urgency": 1, "Emergency": 2}[urgency]
            row = {c: 0.0 for c in feature_cols}
            row['SLA_hours']     = sla_hours
            row['urgency_level'] = urgency_level
            row['month_sin']     = np.sin(2*np.pi*opening_month/12)
            row['month_cos']     = np.cos(2*np.pi*opening_month/12)
            row['dow_sin']       = np.sin(2*np.pi*opening_dow/7)
            row['dow_cos']       = np.cos(2*np.pi*opening_dow/7)
            row['hour_sin']      = np.sin(2*np.pi*opening_hour/24)
            row['hour_cos']      = np.cos(2*np.pi*opening_hour/24)
            mgmt_col = f'mgmt_{management}'
            if mgmt_col in row: row[mgmt_col] = 1.0
            cat_col  = f'cat_{category}'
            if cat_col in row: row[cat_col] = 1.0
            building_idx = sorted(df['ID_Building'].unique()).index(building)
            row['building_code'] = building_idx
            X_input = pd.DataFrame([row])[feature_cols]

            proba = clf_model.predict_proba(X_input)[0, 1]
            pred_delay_log = reg_model.predict(X_input)[0]
            pred_delay = max(0, np.expm1(pred_delay_log) - shift_val)
            rps_value = proba * pred_delay * urgency_mult[urgency_level]

            st.subheader("📊 Prediction Results")
            risk_class = "risk-high" if proba >= 0.7 else "risk-medium" if proba >= 0.4 else "risk-low"
            risk_label = "🔴 HIGH RISK" if proba >= 0.7 else "🟡 MEDIUM RISK" if proba >= 0.4 else "🟢 LOW RISK"

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba * 100,
                title = {'text': "Probability of SLA Breach (%)", 'font': {'size': 18}},
                number = {'suffix': "%", 'font': {'size': 36}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1F4E79"},
                    'steps': [
                        {'range': [0, 40], 'color': "#C8E6C9"},
                        {'range': [40, 70], 'color': "#FFF59D"},
                        {'range': [70, 100], 'color': "#FFCDD2"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""<div class="metric-card {risk_class}">
                <h4 style="margin:0;">{risk_label}</h4>
                <p style="margin:5px 0 0 0; font-size: 24px; font-weight: bold;">{proba*100:.1f}%</p>
                <p style="margin:0; font-size: 12px;">breach probability</p></div>""", unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""<div class="metric-card">
                <h4 style="margin:0;">⏱️ Predicted Delay</h4>
                <p style="margin:5px 0 0 0; font-size: 24px; font-weight: bold;">{pred_delay:.0f}h</p>
                <p style="margin:0; font-size: 12px;">≈ {pred_delay/24:.1f} days beyond SLA</p></div>""", unsafe_allow_html=True)
            with col_m3:
                rps_color = "risk-high" if rps_value > 200 else "risk-medium" if rps_value > 50 else "risk-low"
                st.markdown(f"""<div class="metric-card {rps_color}">
                <h4 style="margin:0;">🎯 RPS Score</h4>
                <p style="margin:5px 0 0 0; font-size: 24px; font-weight: bold;">{rps_value:.0f}</p>
                <p style="margin:0; font-size: 12px;">Risk Priority Score</p></div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📈 Compared to Historical Data")
            similar = df[(df['macro_category'] == category) & (df['urgency_level'] == urgency_level)]
            if len(similar) > 0:
                hist_delay_rate = similar['is_delayed'].mean() * 100
                hist_avg_delay  = similar['delay_hours'].clip(lower=0).mean()
                col_h1, col_h2 = st.columns(2)
                col_h1.metric(f"Historical breach rate ({category} / {urgency})",
                              f"{hist_delay_rate:.1f}%",
                              f"{(proba*100 - hist_delay_rate):+.1f}pp vs prediction")
                col_h2.metric("Historical avg delay", f"{hist_avg_delay:.0f}h",
                              f"{(pred_delay - hist_avg_delay):+.0f}h vs prediction")
                st.caption(f"_Based on {len(similar):,} similar past tickets._")

            st.markdown("---")
            st.subheader("💡 Managerial Recommendation")
            if proba >= 0.7:
                st.error(f"""**🚨 PROACTIVE INTERVENTION REQUIRED**

                High breach risk ({proba*100:.0f}%). Recommended actions:
                - Assign to a senior technician with priority
                - Pre-allocate spare parts to avoid procurement delays
                - Schedule a follow-up checkpoint within {sla_hours/3:.1f}h
                - Consider re-negotiating SLA with the requester""")
            elif proba >= 0.4:
                st.warning(f"""**⚠️ STANDARD MONITORING**

                Moderate risk ({proba*100:.0f}%). Recommended actions:
                - Routine assignment, but flag in the daily report
                - Notify the on-call supervisor if not closed by {sla_hours*0.7:.1f}h""")
            else:
                st.success(f"""**✅ ROUTINE HANDLING**

                Low risk ({proba*100:.0f}%). Standard process applies.
                Expected resolution well within the {sla_hours:.1f}h SLA.""")
        else:
            st.info("👈 Fill in the ticket information on the left and click **Predict Risk** to see the result.")
            st.markdown("---")
            st.markdown("### 🎓 How does this work?")
            st.markdown("""
            This page combines **two XGBoost models** to deliver an instant prediction:

            1. **Classifier** → Probability that the ticket will breach its SLA (AUC = 0.86)
            2. **Regressor** → Expected delay magnitude in hours (R² = 0.46)
            3. **RPS** = Probability × Delay × Urgency Multiplier
               - Above 200 → high priority (top ~10%)
               - 50-200 → medium priority
               - Below 50 → routine handling

            The recommendation is based on our **Prescriptive Analytics** findings:
            **proactive intervention on the top 20% by RPS reduces total portfolio delay by 30%.**
            """)

# ==========================================================================
# PAGE 2 — PORTFOLIO ANALYTICS
# ==========================================================================
elif page == "📊 Portfolio Analytics":
    st.markdown("""<div class="main-header">
        <h1 style="margin:0;">📊 Portfolio Analytics</h1>
        <p style="margin:5px 0 0 0;">Overview of 115 buildings at POLIMI Milano campus</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets", f"{len(df):,}")
    col2.metric("Buildings", df['ID_Building'].nunique())
    col3.metric("Delay Rate", f"{df['is_delayed'].mean()*100:.1f}%")
    col4.metric("Avg Delay", f"{df['delay_hours'].clip(lower=0).mean():.0f}h")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Delay Rate by Urgency")
        urgency_data = df.groupby('Urgency').agg(delay_rate=('is_delayed', 'mean'),
                                                  count=('is_delayed', 'count')).reset_index()
        urgency_data['delay_rate'] *= 100
        urgency_data = urgency_data.sort_values('delay_rate', ascending=False)
        fig = px.bar(urgency_data, x='Urgency', y='delay_rate',
                     text=urgency_data['delay_rate'].apply(lambda x: f'{x:.1f}%'),
                     color='delay_rate', color_continuous_scale='Reds',
                     labels={'delay_rate': 'Delay Rate (%)'})
        fig.update_layout(showlegend=False, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Delay Rate by Macro-Category")
        cat_data = df.groupby('macro_category').agg(delay_rate=('is_delayed', 'mean'),
                                                     count=('is_delayed', 'count')).reset_index()
        cat_data = cat_data[cat_data['count'] >= 100]
        cat_data['delay_rate'] *= 100
        cat_data = cat_data.sort_values('delay_rate', ascending=True)
        fig = px.bar(cat_data, y='macro_category', x='delay_rate', orientation='h',
                     text=cat_data['delay_rate'].apply(lambda x: f'{x:.0f}%'),
                     color='delay_rate', color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Trend")
    monthly = df.groupby('opening_month').agg(tickets=('is_delayed', 'count'),
                                                delay_rate=('is_delayed', 'mean')).reset_index()
    monthly['delay_rate'] *= 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly['opening_month'], y=monthly['tickets'],
                         name='Tickets', marker_color='#1F4E79', yaxis='y1'))
    fig.add_trace(go.Scatter(x=monthly['opening_month'], y=monthly['delay_rate'],
                             name='Delay Rate (%)', line=dict(color='#F44336', width=3),
                             marker=dict(size=10), yaxis='y2'))
    fig.update_layout(height=380, xaxis=dict(title='Month', tickmode='linear'),
                      yaxis=dict(title='Number of Tickets'),
                      yaxis2=dict(title='Delay Rate (%)', overlaying='y', side='right', range=[0, 100]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 Buildings by Volume")
    top_buildings = df.groupby('ID_Building').agg(tickets=('is_delayed', 'count'),
                                                    delay_rate=('is_delayed', 'mean')).reset_index().sort_values('tickets', ascending=False).head(15)
    top_buildings['delay_rate'] *= 100
    top_buildings['ID_Building'] = top_buildings['ID_Building'].astype(str)
    fig = px.bar(top_buildings, x='ID_Building', y='tickets',
                 color='delay_rate', color_continuous_scale='RdYlGn_r',
                 labels={'tickets': 'Number of Tickets', 'delay_rate': 'Delay Rate (%)'})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# PAGE 3 — BUILDING DEEP DIVE
# ==========================================================================
elif page == "🏢 Building Deep Dive":
    st.markdown("""<div class="main-header">
        <h1 style="margin:0;">🏢 Building Deep Dive</h1>
        <p style="margin:5px 0 0 0;">Per-building diagnostic analytics</p>
    </div>""", unsafe_allow_html=True)

    buildings_sorted = df['ID_Building'].value_counts().index.tolist()
    selected_building = st.selectbox("Select a building (sorted by ticket volume):",
        buildings_sorted,
        format_func=lambda x: f"Building {x} ({df[df['ID_Building']==x].shape[0]} tickets)")
    bdf = df[df['ID_Building'] == selected_building]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets", len(bdf))
    col2.metric("Delay Rate", f"{bdf['is_delayed'].mean()*100:.1f}%",
                f"{(bdf['is_delayed'].mean() - df['is_delayed'].mean())*100:+.1f}pp vs portfolio")
    col3.metric("Avg Delay", f"{bdf['delay_hours'].clip(lower=0).mean():.0f}h",
                f"{(bdf['delay_hours'].clip(lower=0).mean() - df['delay_hours'].clip(lower=0).mean()):+.0f}h vs portfolio")
    col4.metric("Avg RPS", f"{bdf['RPS'].mean():.0f}",
                f"{(bdf['RPS'].mean() - df['RPS'].mean()):+.0f} vs portfolio")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"Categories at Building {selected_building}")
        cat_data = bdf.groupby('macro_category').agg(tickets=('is_delayed', 'count'),
                                                       delay_rate=('is_delayed', 'mean')).reset_index().sort_values('tickets', ascending=False).head(8)
        cat_data['delay_rate'] *= 100
        fig = px.bar(cat_data, x='macro_category', y='tickets',
                     color='delay_rate', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader(f"Performance by Contractor")
        mgmt_data = bdf.groupby('Management').agg(tickets=('is_delayed', 'count'),
                                                    delay_rate=('is_delayed', 'mean')).reset_index()
        mgmt_data = mgmt_data[mgmt_data['tickets'] >= 5].sort_values('delay_rate', ascending=True)
        mgmt_data['delay_rate'] *= 100
        if len(mgmt_data) > 0:
            fig = px.bar(mgmt_data, y='Management', x='delay_rate', orientation='h',
                         text=mgmt_data['delay_rate'].apply(lambda x: f'{x:.0f}%'),
                         color='delay_rate', color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for contractor analysis.")

    st.markdown("---")
    st.subheader(f"⚠️ Top 10 Highest-Risk Past Tickets at Building {selected_building}")
    top_risk = bdf.nlargest(10, 'RPS')[['Urgency', 'macro_category', 'Management',
        'SLA_hours', 'delay_hours', 'pred_proba', 'pred_delay_hours', 'RPS']].copy()
    top_risk.columns = ['Urgency', 'Category', 'Contractor', 'SLA (h)',
                        'Actual Delay (h)', 'Pred. Prob.', 'Pred. Delay (h)', 'RPS']
    top_risk['Pred. Prob.'] = (top_risk['Pred. Prob.'] * 100).round(1).astype(str) + '%'
    top_risk['Pred. Delay (h)'] = top_risk['Pred. Delay (h)'].round(0)
    top_risk['RPS'] = top_risk['RPS'].round(0)
    st.dataframe(top_risk, use_container_width=True)
