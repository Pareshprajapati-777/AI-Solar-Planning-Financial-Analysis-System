import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import io
import base64

st.set_page_config(page_title="AI Solar Planning & Financial System", page_icon="☀️", layout="wide")

custom_css = """
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    .animated-card {
        animation: fadeIn 0.5s ease-in-out;
        padding: 15px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e293b, #334155);
        text-align: center;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        border: 1px solid #4f46e5;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 5px;
    }
    .main-header {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #172554, #312e81);
        color: white;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0,0,0,0.4);
        margin-bottom: 25px;
        border: 1px solid #6366f1;
    }
    div.stTabs > div.stTabs[aria-busy="true"]::before {
        display: none;
    }
    div.stTabs > div.stTabs > button {
        background-color: #1e293b;
        color: white;
        border: 1px solid #4f46e5;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    div.stTabs > div.stTabs > button[data-baseweb="tab-selected"] {
        background-color: #4f46e5;
        color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #4338ca, #4f46e5);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
        transform: translateY(-2px);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize Session State for History and KPIs
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=['Time', 'Irradiation', 'Module Temp', 'Ambient Temp', 'Hour', 'Month', 'Predicted DC Power'])
if 'planner_history' not in st.session_state:
    st.session_state.planner_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = 0.0

@st.cache_data
def load_data():
    return pd.read_csv("Solar_final.csv")

@st.cache_resource
def train_model(df):
    df.drop_duplicates(inplace=True)
    X = df.drop(["DC_POWER", "DATE_TIME"], axis=1)
    y = df["DC_POWER"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train_poly, y_train)
    prediction = model.predict(X_test_poly)
    return model, X, poly, y_test, prediction

df = load_data()
model, X, poly, y_test, prediction = train_model(df)

sco = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
rmse = mean_squared_error(y_test, prediction) ** 0.5

st.title("☀️ AI Solar Planning & Financial Analysis System")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Data Visualization", "🔮 Prediction Hub", "ℹ️ About"])

with tab1:
    st.markdown('<div class="main-header"><h2>🌞 Enterprise Solar Analytics Dashboard</h2><p>Real-time insights, ML Predictions, and Financial Metrics</p></div>', unsafe_allow_html=True)
    
    last_pred = st.session_state.last_prediction
    
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">⚡</div><div class="metric-value">{last_pred:.2f} kW</div><div class="metric-label">Last Predicted Power</div></div>', unsafe_allow_html=True)
    with col2:
        eff = (last_pred / 150) * 100 if last_pred > 0 else 0           
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">🧠</div><div class="metric-value">{eff:.1f}%</div><div class="metric-label">System Efficiency</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">💰</div><div class="metric-value">₹15,420</div><div class="metric-label">Est. Monthly Savings</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">🌱</div><div class="metric-value">1.2 Tons</div><div class="metric-label">CO₂ Saved Monthly</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4, gap="small")
    with col5:
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">📈</div><div class="metric-value">5.2 Yrs</div><div class="metric-label">Avg ROI Period</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">🎯</div><div class="metric-value">{sco:.3f}</div><div class="metric-label">ML Model R² Score</div></div>', unsafe_allow_html=True)
    with col7:
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">🧊</div><div class="metric-value">{len(df):,}</div><div class="metric-label">Data Points Analyzed</div></div>', unsafe_allow_html=True)
    with col8:
        st.markdown(f'<div class="animated-card"><div style="font-size:24px;">🌳</div><div class="metric-value">54</div><div class="metric-label">Equivalent Trees</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")
    st.success("✅ System Online: Decision Tree Regressor Model Loaded Successfully")

with tab2:
    st.markdown('<div class="main-header"><h2>📊 Historical Solar Data Trends</h2><p>Deep dive into 2020 Plant Data</p></div>', unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("⏰ Avg DC Power by Hour")
        hourly = df.groupby("HOUR")["DC_POWER"].mean().reset_index()
        hourly.columns = ["Hour", "Avg DC Power"]
        fig_hour = px.bar(hourly, x="Hour", y="Avg DC Power", color="Avg DC Power", color_continuous_scale="Blues", text_auto=".0f")
        fig_hour.update_layout(showlegend=False, coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col_chart2:
        st.subheader("📅 Monthly Average DC Power")
        monthly = df.groupby("MONTH")["DC_POWER"].mean().reset_index()
        monthly["Month"] = monthly["MONTH"].map({4: "April", 5: "May", 6: "June", 7: "July"})
        fig_month = px.pie(monthly, values="DC_POWER", names="Month", hole=0.4, color_discrete_sequence=px.colors.sequential.Blues)
        fig_month.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")
    st.subheader("☀️ Irradiation vs DC Power (Sample)")
    sample_df = df[df["DC_POWER"] > 0].sample(2000, random_state=42)
    fig_scatter = px.scatter(sample_df, x="IRRADIATION", y="DC_POWER", color="MODULE_TEMPERATURE", color_continuous_scale="RdYlGn_r", opacity=0.7)
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown('<div class="main-header"><h2>🔮 AI Prediction & Planning Hub</h2><p>Choose your analysis mode below</p></div>', unsafe_allow_html=True)
    
    option1, option2, option3 = st.tabs(["🏠 Home Solar Planner", "☀ Existing Plant ML Prediction", "🆕 New Commercial/Industrial Planner"])

    with option1:
        st.markdown("### 🏠 Get Complete Solar Financial & Technical Plan")
        with st.expander("⚙️ Enter Your Details", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                state = st.text_input("State", "Maharashtra")
                city = st.text_input("City", "Pune")
                roof_type = st.selectbox("Roof Type", ["Flat (Concrete)", "Sloped (Tin/Shed)"])
                roof_area = st.number_input("Roof Area (sq.ft)", min_value=100, max_value=10000, value=1500)
                monthly_bill = st.number_input("Monthly Electricity Bill (₹)", min_value=500, max_value=50000, value=5000)
            with c2:
                monthly_units = st.number_input("Monthly Consumption (Units/kWh)", min_value=50, max_value=5000, value=300)
                house_type = st.selectbox("House Type", ["1 BHK", "2 BHK", "3 BHK", "Villa/Bungalow"])
                tariff = st.number_input("Electricity Tariff (₹/Unit)", min_value=5.0, max_value=20.0, value=8.5, step=0.5)
                include_battery = st.checkbox("Include Battery Backup?", value=False)
            
            plan_btn = st.button("🚀 Generate Complete Solar Plan", use_container_width=True)

        if plan_btn:
            with st.spinner("🧠 AI is calculating optimal system size, subsidies, and ROI..."):
                # --- CALCULATIONS ---
                # 1. System Sizing (Assumption: 4 Peak Sun Hours, 30 days)
                req_capacity_kw = (monthly_units / 30) / 4.0 
                # Adjust for roof area constraint (1 kW requires ~100 sqft)
                max_capacity_by_roof = roof_area / 100.0
                final_capacity = min(req_capacity_kw, max_capacity_by_roof)
                final_capacity = max(1.0, round(final_capacity, 2)) # Minimum 1kW

                panel_watt = 540 # Modern Monocrystalline
                num_panels = int(np.ceil((final_capacity * 1000) / panel_watt))
                area_required = num_panels * 22 # approx 22 sqft per 540W panel
                
                # Inverter sizing (1.2x capacity)
                inverter_kw = final_capacity * 1.2
                if inverter_kw <= 3: inverter_type = f"{round(inverter_kw, 1)} kVA String Inverter"
                elif inverter_kw <= 5: inverter_type = f"{round(inverter_kw, 1)} kVA String Inverter"
                else: inverter_type = f"{round(inverter_kw, 1)} kVA String/Hybrid Inverter"

                # Generation
                daily_gen = final_capacity * 4.0
                monthly_gen = daily_gen * 30
                annual_gen = daily_gen * 365

                # Financials
                cost_per_kw = 45000 # Avg current market rate for 540W panels
                panel_cost = final_capacity * cost_per_kw
                installation_cost = panel_cost * 0.12
                inverter_cost = final_capacity * 7000
                battery_cost = final_capacity * 25000 if include_battery else 0
                total_cost = panel_cost + installation_cost + inverter_cost + battery_cost

                # PM Surya Ghar Subsidy Logic
                if final_capacity <= 2:
                    subsidy = final_capacity * 30000
                elif final_capacity <= 3:
                    subsidy = (2 * 30000) + ((final_capacity - 2) * 18000)
                else:
                    subsidy = (2 * 30000) + (1 * 18000) # Max cap at 3kW subsidy
                subsidy = min(subsidy, 78000) # Absolute cap
                
                final_cost = total_cost - subsidy

                # ROI & Savings
                monthly_savings = min(monthly_gen, monthly_units) * tariff
                yearly_savings = monthly_savings * 12
                payback_years = final_cost / yearly_savings if yearly_savings > 0 else 99
                
                maintenance_yearly = 2000
                lifetime_savings_25 = (yearly_savings * 25) - (maintenance_yearly * 25)
                net_profit = lifetime_savings_25 - final_cost

                # Environment
                co2_saved_annual = annual_gen * 0.7 # 0.7kg CO2 per kWh
                trees_equiv = co2_saved_annual / 20 # 1 tree absorbs ~20kg/year

                # --- DISPLAY RESULTS ---
                st.success("✅ Solar Plan Generated Successfully!")
                
                # System Specs
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Recommended Capacity", f"{final_capacity} kW")
                col_s2.metric("Number of Panels", f"{num_panels} Panels ({panel_watt}W)")
                col_s3.metric("Area Required", f"{area_required} sq.ft")
                
                # Financial Cards
                st.markdown("#### 💰 Financial & Subsidy Analysis")
                fin_c1, fin_c2, fin_c3, fin_c4 = st.columns(4)
                fin_c1.markdown(f'<div class="animated-card"><div class="metric-label">Total Project Cost</div><div class="metric-value">₹{total_cost:,.0f}</div></div>', unsafe_allow_html=True)
                fin_c2.markdown(f'<div class="animated-card"><div class="metric-label">Govt Subsidy</div><div class="metric-value" style="color:#34d399;">- ₹{subsidy:,.0f}</div></div>', unsafe_allow_html=True)
                fin_c3.markdown(f'<div class="animated-card"><div class="metric-label">Final Cost to Pay</div><div class="metric-value" style="color:#fbbf24;">₹{final_cost:,.0f}</div></div>', unsafe_allow_html=True)
                fin_c4.markdown(f'<div class="animated-card"><div class="metric-label">Payback Period</div><div class="metric-value">{payback_years:.1f} Yrs</div></div>', unsafe_allow_html=True)

                # 25 Year Profit
                st.markdown("#### 📈 25-Year Lifetime Projection")
                prof_c1, prof_c2, prof_c3 = st.columns(3)
                prof_c1.metric("Total Savings (25 Yrs)", f"₹{yearly_savings*25:,.0f}")
                prof_c2.metric("Maintenance Cost (25 Yrs)", f"- ₹{maintenance_yearly*25:,.0f}")
                prof_c3.metric("Net Lifetime Profit", f"₹{net_profit:,.0f}", delta_color="off")

                # Monthly Profit Table & Chart
                st.markdown("#### 📅 Monthly Profit Analysis (Year 1)")
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                # Simulate seasonal variation (Multiplier: Winter low, Summer high)
                seasonal_factors = [0.85, 0.95, 1.1, 1.2, 1.25, 1.15, 1.0, 0.95, 1.0, 1.1, 0.9, 0.8]
                
                monthly_data = []
                for i, m in enumerate(months):
                    gen = daily_gen * 30 * seasonal_factors[i]
                    bill_before = monthly_bill
                    savings = min(gen, monthly_units) * tariff
                    bill_after = max(0, bill_before - savings)
                    monthly_data.append({
                        "Month": m, 
                        "Generated (kWh)": round(gen, 1), 
                        "Bill Before (₹)": round(bill_before, 0),
                        "Bill After (₹)": round(bill_after, 0), 
                        "Savings (₹)": round(savings, 0)
                    })
                
                df_monthly = pd.DataFrame(monthly_data)
                st.dataframe(df_monthly, use_container_width=True, hide_index=True)
                
                fig_profit = go.Figure()
                fig_profit.add_trace(go.Bar(x=df_monthly["Month"], y=df_monthly["Savings (₹)"], name="Savings", marker_color='#6366f1'))
                fig_profit.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", title="Monthly Savings Trend")
                st.plotly_chart(fig_profit, use_container_width=True)

                # Environment Impact
                st.markdown("#### 🌱 Environmental Impact")
                env_c1, env_c2, env_c3 = st.columns(3)
                env_c1.markdown(f'<div class="animated-card"><div style="font-size:30px;">🌍</div><div class="metric-value">{co2_saved_annual/1000:.1f} Tons</div><div class="metric-label">CO₂ Saved Annually</div></div>', unsafe_allow_html=True)
                env_c2.markdown(f'<div class="animated-card"><div style="font-size:30px;">🌳</div><div class="metric-value">{trees_equiv:.0f} Trees</div><div class="metric-label">Equivalent Trees Planted</div></div>', unsafe_allow_html=True)
                env_c3.markdown(f'<div class="animated-card"><div style="font-size:30px;">🔋</div><div class="metric-value">{annual_gen/1000:.1f} MWh</div><div class="metric-label">Green Energy Generated</div></div>', unsafe_allow_html=True)

                # --- PDF GENERATION ---
                def create_pdf_report(data_dict):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", size=12)
                    
                    # Header
                    pdf.set_fill_color(30, 41, 59) # Dark Blue
                    pdf.rect(0, 0, 210, 40, 'F')
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_font("Helvetica", 'B', 20)
                    pdf.cell(0, 20, 'AI Solar Planning Report', 0, 1, 'C')
                    pdf.set_font("Helvetica", size=10)
                    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                    
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(10)
                    
                    # Sections
                    pdf.set_font("Helvetica", 'B', 14)
                    pdf.cell(0, 10, '1. User Inputs', 0, 1)
                    pdf.set_font("Helvetica", size=11)
                    pdf.multi_cell(0, 6, f"City: {city}, {state}\nRoof Area: {roof_area} sq.ft\nMonthly Bill: Rs {monthly_bill}\nConsumption: {monthly_units} Units")
                    
                    pdf.ln(5)
                    pdf.set_font("Helvetica", 'B', 14)
                    pdf.cell(0, 10, '2. Recommended System', 0, 1)
                    pdf.set_font("Helvetica", size=11)
                    pdf.multi_cell(0, 6, f"Capacity: {final_capacity} kW\nPanels: {num_panels} x {panel_watt}W\nInverter: {inverter_type}\nArea Required: {area_required} sq.ft")
                    
                    pdf.ln(5)
                    pdf.set_font("Helvetica", 'B', 14)
                    pdf.cell(0, 10, '3. Financial Analysis', 0, 1)
                    pdf.set_font("Helvetica", size=11)
                    pdf.multi_cell(0, 6, f"Total Cost: Rs {total_cost:,.0f}\nGovt Subsidy: Rs {subsidy:,.0f}\nFinal Cost: Rs {final_cost:,.0f}\nPayback: {payback_years:.1f} Years\nNet 25-Yr Profit: Rs {net_profit:,.0f}")

                    return pdf.output(dest="S").encode("latin-1")

                pdf_data = create_pdf_report(locals())
                st.download_button(
                    label="📄 Download Full PDF Report",
                    data=pdf_data,
                    file_name=f"Solar_Plan_{city}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

    with option2:
        st.markdown("### ☀ Predict DC Power for Existing Plant")
        st.caption("Using Decision Tree Regressor trained on 50,000+ data points.")

        st.subheader("📖 Feature Description")
        st.dataframe(pd.DataFrame({"Feature": ["☀️ Irradiation", "🌡️ Module Temperature", "🌤️ Ambient Temperature", "🕒 Hour", "📅 Month"], "Description": ["Solar panel par girne wali sunlight ki intensity.", "Solar panel ka actual temperature.", "Bahar ke environment ka temperature.", "Din ka samay (0–23).", "Saal ka mahina (4–7)."], "Unit": ["kW/m²", "°C", "°C", "0–23", "4–7"], "Effect": ["Higher irradiation → Higher electricity generation.", "High temperature reduces efficiency.", "Indirectly affects panel temperature.", "Peak generation around 10 AM–4 PM.", "May sabse productive month hai."]}), use_container_width=True, hide_index=True)
        st.info("⚡ **Target Variable (DC_POWER):** Solar panel dwara utpann Direct Current (DC) electrical power.")
        st.header("🔮 Solar Power Prediction")
        with st.form("prediction_form"):
            c_in1, c_in2 = st.columns(2)
            with c_in1:
                irradiation = st.number_input("☀️ Irradiation (kW/m²)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
                module_temp = st.number_input("🌡️ Module Temperature (°C)", min_value=15.0, max_value=70.0, value=35.0)
            with c_in2:
                ambient_temp = st.number_input("🌤️ Ambient Temperature (°C)", min_value=15.0, max_value=45.0, value=27.0)
                c_hr, c_mn = st.columns(2)
                hour = c_hr.slider("🕒 Hour", 0, 23, 12)
                month = c_mn.slider("📅 Month", 4, 7, 5)
            
            pred_btn = st.form_submit_button("🚀 Run ML Prediction", use_container_width=True)

        if pred_btn:
            # Feature Engineering matching training data
            sample = pd.DataFrame([{
                "PLANT": df["PLANT"].mode()[0],
                "HOUR": hour, "MINUTE": 0, "DAY_OF_WEEK": 3, "DAY_OF_MONTH": 15,
                "MONTH": month, "TIME_OF_DAY": hour + 0 / 60.0,
                "HOUR_SIN": np.sin(2 * np.pi * hour / 24), "HOUR_COS": np.cos(2 * np.pi * hour / 24),
                "IRRADIATION": irradiation, "IRRAD_SQUARED": irradiation**2,
                "AMBIENT_TEMPERATURE": ambient_temp, "MODULE_TEMPERATURE": module_temp,
                "TEMP_DIFF": module_temp - ambient_temp,
                "IRRAD_X_MODULE_TEMP": irradiation * module_temp,
                "IS_DAYTIME": 1 if 6 <= hour <= 18 else 0,
                "IRRAD_LAG1": irradiation
            }])
            sample = sample[X.columns]
            
            ans = max(0.0, model.predict(poly.transform(sample))[0])
            st.session_state.last_prediction = ans # Update Dashboard KPI
            
            # History Management
            new_hist = pd.DataFrame([{
                'Time': datetime.now().strftime("%H:%M:%S"),
                'Irradiation': irradiation,
                'Module Temp': module_temp,
                'Ambient Temp': ambient_temp,
                'Hour': hour,
                'Month': month,
                'Predicted DC Power': ans
            }])
            st.session_state.prediction_history = pd.concat([st.session_state.prediction_history, new_hist], ignore_index=True)

            # Display Results
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.markdown(f'<div class="animated-card"><div style="font-size:30px;">⚡</div><div class="metric-value">{ans:.2f} kW</div><div class="metric-label">Predicted DC Power</div></div>', unsafe_allow_html=True)
            
            # Efficiency
            eff = (ans / (irradiation * 1000 + 0.001)) * 100
            res_c2.markdown(f'<div class="animated-card"><div style="font-size:30px;">🎯</div><div class="metric-value">{eff:.1f}%</div><div class="metric-label">Conversion Efficiency</div></div>', unsafe_allow_html=True)
            
            # Performance vs Historical
            hist_avg = df[(df["HOUR"] == hour) & (df["MONTH"] == month)]["DC_POWER"].mean()
            perf_delta = ((ans - hist_avg) / hist_avg) * 100 if hist_avg > 0 else 0
            res_c3.markdown(f'<div class="animated-card"><div style="font-size:30px;">📊</div><div class="metric-value">{perf_delta:+.1f}%</div><div class="metric-label">vs Historical Avg</div></div>', unsafe_allow_html=True)

            # Plotly Gauge Meter
            st.markdown("#### ⚡ Power Gauge Meter")
            gauge_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ans,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "DC Power Output (kW)", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 150], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                    'bar': {'color': "#4f46e5"},
                    'steps': [
                        {'range': [0, 50], 'color': '#1e293b'},
                        {'range': [50, 100], 'color': '#334155'},
                        {'range': [100, 150], 'color': '#475569'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': hist_avg
                    }
                }
            ))
            gauge_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

            # AI Insights
            st.markdown("#### 🧠 AI Insights & Maintenance Advice")
            ins_c1, ins_c2 = st.columns(2)
            with ins_c1:
                st.info(f"**Historical Context:** Average power for Hour {hour} in Month {month} is **{hist_avg:.2f} kW**.")
                risk = "Low" if eff > 15 else "Medium" if eff > 10 else "High"
                risk_color = "🟢" if risk=="Low" else "🟡" if risk=="Medium" else "🔴"
                st.warning(f"{risk_color} **Risk Level:** {risk} (Based on efficiency metrics)")
                
            with ins_c2:
                suggestions = []
                if module_temp > 45: suggestions.append("⚠️ Module temp high. Check for dust or cooling airflow issues.")
                if irradiation < 0.2 and hour > 8: suggestions.append("☁️ Low irradiation. Likely cloudy weather or soiling.")
                if perf_delta < -10: suggestions.append("📉 Underperforming vs history. Inspect for shading or inverter clipping.")
                if not suggestions: suggestions.append("✅ System operating optimally according to ML model parameters.")
                
                for s in suggestions:
                    st.markdown(s)

        # Prediction History Section
        st.markdown("---")
        st.subheader("📜 Prediction History")
        if not st.session_state.prediction_history.empty:
            st.dataframe(st.session_state.prediction_history.tail(10), use_container_width=True)
            csv = st.session_state.prediction_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export History as CSV",
                data=csv,
                file_name='solar_ml_predictions.csv',
                mime='text/csv'
            )
        else:
            st.info("No predictions made yet in this session.")


    with option3:
        st.markdown("### 🆕 Design a New Commercial / Industrial Solar Plant")
        st.caption("End-to-end capacity design, cost estimation, land analysis, and 25-year financial simulation for new solar projects.")

        with st.expander("⚙️ Project Details", expanded=True):
            st.subheader("📖 Option Guide")

            info_df = pd.DataFrame({
                "Option": [
                    "🔋 Battery Storage Required",
                    "⚡ Grid Connected",
                    "🔄 Net Metering Required"
                ],
                "Meaning": [
                    "Stores extra solar energy in batteries for use during power cuts or at night.",
                    "Connects the solar plant to the electricity grid for continuous power supply.",
                    "Sends extra solar electricity to the grid and provides bill credits for exported energy."
                ],
                "When to Select": [
                    "Choose if backup power is needed or power cuts are frequent.",
                    "Recommended for most residential, commercial, and industrial solar systems.",
                    "Choose if your local electricity provider supports net metering."
                ],
                "Recommended": [
                    "Only if backup is required",
                    "✅ Yes",
                    "✅ Yes"
                ]
            })

            st.dataframe(
                info_df,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "💡 **Recommended for most Commercial & Industrial Projects:** "
                "Grid Connected ✅ + Net Metering ✅. "
                "Battery Storage is optional and mainly required for backup during power outages."
            )
            
            n_c1, n_c2, n_c3 = st.columns(3)
            with n_c1:
                plant_type_new = st.selectbox(
                    "Plant Type",
                    ["Commercial", "Industrial", "Residential", "Institution", "Factory", "Warehouse", "Farm", "Hotel", "Hospital", "School", "College"],
                    key="m3_plant_type"
                )
                land_unit = st.selectbox("Land Area Unit", ["Square Feet", "Acres"], key="m3_land_unit")
                land_area_input = st.number_input(
                    f"Available Land Area ({land_unit})",
                    min_value=1.0,
                    value=1.0 if land_unit == "Acres" else 5000.0,
                    step=0.5 if land_unit == "Acres" else 100.0,
                    key="m3_land_area"
                )
            with n_c2:
                land_state = st.text_input("State", "Gujarat", key="m3_state")
                land_city = st.text_input("City", "Ahmedabad", key="m3_city")
                monthly_units_new = st.number_input("Monthly Electricity Consumption (kWh)", min_value=500, max_value=2000000, value=20000, key="m3_units")
                monthly_bill_new = st.number_input("Monthly Electricity Bill (₹)", min_value=5000, max_value=20000000, value=180000, key="m3_bill")
            with n_c3:
                budget_new = st.number_input("Available Budget (₹)", min_value=100000, max_value=500000000, value=15000000, step=100000, key="m3_budget")
                tariff_new = st.number_input("Electricity Tariff (₹/Unit)", min_value=5.0, max_value=20.0, value=9.0, step=0.5, key="m3_tariff")
                battery_required_new = st.checkbox("Battery Storage Required?", value=False, key="m3_battery")
                grid_connected_new = st.checkbox("Grid Connected?", value=True, key="m3_grid")
                net_metering_new = st.checkbox("Net Metering Required?", value=True, key="m3_netmeter")

            design_btn = st.button("🚀 Design Complete Solar Plant", use_container_width=True, key="m3_design_btn")

        if design_btn:
            with st.spinner("🧠 AI is designing the plant, estimating cost, and running 25-year simulation..."):

                # ---------------- LAND CONVERSION ----------------
                sqft_per_acre = 43560
                land_available_sqft = land_area_input * sqft_per_acre if land_unit == "Acres" else land_area_input

                # ---------------- CAPACITY SIZING ----------------
                sqft_per_kw_ground_mount = 130  # ground-mounted with row spacing needs more area than rooftop
                cost_per_kw_equipment = 42000    # blended panel+structure+inverter rate for bulk commercial projects

                capacity_by_consumption = (monthly_units_new / 30) / 4.0
                capacity_by_land = land_available_sqft / sqft_per_kw_ground_mount
                capacity_by_budget = budget_new / (cost_per_kw_equipment * 1.25)  # 1.25x buffer for civil+GST+misc

                final_capacity_new = min(capacity_by_consumption, capacity_by_land, capacity_by_budget)
                final_capacity_new = max(5.0, round(final_capacity_new, 1))  # minimum 5kW for a "new plant" project

                panel_watt_new = 590  # Bifacial Mono PERC, standard for commercial/industrial
                num_panels_new = int(np.ceil((final_capacity_new * 1000) / panel_watt_new))
                land_required_sqft = num_panels_new * (sqft_per_kw_ground_mount * panel_watt_new / 1000)
                remaining_land_sqft = max(0.0, land_available_sqft - land_required_sqft)
                land_utilization_pct = min(100.0, (land_required_sqft / land_available_sqft) * 100) if land_available_sqft > 0 else 0
                max_possible_capacity = land_available_sqft / sqft_per_kw_ground_mount

                # Panel / Inverter / Structure recommendation
                panel_brand_new = "Waaree / Adani Solar / Tata Power Solar"
                panel_technology_new = "Bifacial Mono PERC" if final_capacity_new > 100 else "Mono PERC"
                if final_capacity_new <= 50:
                    inverter_capacity_new = f"{round(final_capacity_new * 1.1, 1)} kW String Inverter"
                elif final_capacity_new <= 500:
                    inverter_capacity_new = f"{round(final_capacity_new * 1.1, 1)} kW String Inverter Cluster"
                else:
                    inverter_capacity_new = f"{round(final_capacity_new * 1.1, 1)} kW Central Inverter"
                transformer_required = final_capacity_new > 500
                structure_type_new = "Ground-Mounted MMS (Mild Steel/GI Structure)"
                cable_length_est_m = round(np.sqrt(land_required_sqft) * 4, 0)  # rough perimeter+run estimate

                battery_capacity_kwh = round(final_capacity_new * 2, 1) if battery_required_new else 0  # ~2 hrs autonomy

                # ---------------- GENERATION ESTIMATE ----------------
                peak_sun_hours = 4.5
                hourly_gen_new = round(final_capacity_new * 0.85, 2)  # avg output during effective sun hours
                daily_gen_new = final_capacity_new * peak_sun_hours
                monthly_gen_new = daily_gen_new * 30
                annual_gen_new = daily_gen_new * 365
                gen_25yr = sum(annual_gen_new * ((1 - 0.005) ** yr) for yr in range(25))  # 0.5%/yr degradation

                # ---------------- COST ESTIMATION ----------------
                panel_cost_new = final_capacity_new * 26000
                structure_cost_new = final_capacity_new * 4500
                inverter_cost_new = final_capacity_new * 5500
                transformer_cost_new = final_capacity_new * 1200 if transformer_required else 0
                battery_cost_new = battery_capacity_kwh * 20000
                cabling_cost_new = final_capacity_new * 1500
                installation_cost_new = (panel_cost_new + structure_cost_new + inverter_cost_new) * 0.08
                civil_work_cost_new = final_capacity_new * 2000
                subtotal_new = (panel_cost_new + structure_cost_new + inverter_cost_new + transformer_cost_new
                                 + battery_cost_new + cabling_cost_new + installation_cost_new + civil_work_cost_new)
                gst_new = subtotal_new * 0.138  # blended GST rate for solar equipment + services
                maintenance_annual_new = final_capacity_new * 700
                total_project_cost_new = subtotal_new + gst_new

                # ---------------- SUBSIDY ----------------
                if plant_type_new == "Residential":
                    if final_capacity_new <= 2:
                        subsidy_new = final_capacity_new * 30000
                    elif final_capacity_new <= 3:
                        subsidy_new = (2 * 30000) + ((final_capacity_new - 2) * 18000)
                    else:
                        subsidy_new = (2 * 30000) + (1 * 18000)
                    subsidy_new = min(subsidy_new, 78000)
                    subsidy_note = "PM Surya Ghar Muft Bijli Yojana subsidy applied (residential only)."
                else:
                    subsidy_new = 0
                    subsidy_note = "Direct capital subsidy typically not applicable for Commercial/Industrial plants. Benefits may include Accelerated Depreciation (up to 40%) and state-specific policy incentives — please verify current policy for your state."

                final_cost_new = total_project_cost_new - subsidy_new

                # ---------------- FINANCIAL ANALYSIS ----------------
                bill_before_new = monthly_bill_new
                monthly_saving_new = min(monthly_gen_new, monthly_units_new) * tariff_new if (grid_connected_new or True) else 0
                if not net_metering_new:
                    monthly_saving_new = min(monthly_gen_new, monthly_units_new) * tariff_new * 0.9  # slight haircut for self-consumption-only setups
                bill_after_new = max(0.0, bill_before_new - monthly_saving_new)
                annual_saving_new = monthly_saving_new * 12
                payback_years_new = final_cost_new / annual_saving_new if annual_saving_new > 0 else 99
                roi_pct_new = (annual_saving_new / final_cost_new) * 100 if final_cost_new > 0 else 0
                lifetime_savings_25_new = (annual_saving_new * 25) - (maintenance_annual_new * 25)
                net_lifetime_profit_new = lifetime_savings_25_new - final_cost_new

                # ---------------- ENVIRONMENT ----------------
                co2_saved_annual_new = annual_gen_new * 0.7
                trees_equiv_new = co2_saved_annual_new / 20

                # ==================== DISPLAY: SYSTEM DESIGN ====================
                st.success("✅ Complete Solar Plant Design Generated Successfully!")

                st.markdown("#### 🏗️ Recommended Plant Design")
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                d_c1.markdown(f'<div class="animated-card"><div class="metric-label">Plant Capacity</div><div class="metric-value">{final_capacity_new} kW</div></div>', unsafe_allow_html=True)
                d_c2.markdown(f'<div class="animated-card"><div class="metric-label">Number of Panels</div><div class="metric-value">{num_panels_new}</div></div>', unsafe_allow_html=True)
                d_c3.markdown(f'<div class="animated-card"><div class="metric-label">Panel Wattage</div><div class="metric-value">{panel_watt_new} W</div></div>', unsafe_allow_html=True)
                d_c4.markdown(f'<div class="animated-card"><div class="metric-label">Panel Technology</div><div class="metric-value" style="font-size:15px;">{panel_technology_new}</div></div>', unsafe_allow_html=True)
                st.write(" ")
                d2_c1, d2_c2, d2_c3, d2_c4 = st.columns(4)
                d2_c1.markdown(f'<div class="animated-card"><div class="metric-label">Inverter</div><div class="metric-value" style="font-size:14px;">{inverter_capacity_new}</div></div>', unsafe_allow_html=True)
                d2_c2.markdown(f'<div class="animated-card"><div class="metric-label">Transformer Needed</div><div class="metric-value">{"Yes" if transformer_required else "No"}</div></div>', unsafe_allow_html=True)
                d2_c3.markdown(f'<div class="animated-card"><div class="metric-label">Battery Capacity</div><div class="metric-value">{battery_capacity_kwh} kWh</div></div>', unsafe_allow_html=True)
                d2_c4.markdown(f'<div class="animated-card"><div class="metric-label">Est. Cable Length</div><div class="metric-value">{cable_length_est_m:,.0f} m</div></div>', unsafe_allow_html=True)
                st.write(" ")
                st.info(f"**Structure Type:** {structure_type_new} | **Panel Brand:** {panel_brand_new}")

                # ==================== DISPLAY: COST ESTIMATION ====================
                st.markdown("#### 💰 Project Cost Estimation")
                cost_df = pd.DataFrame({
                    "Component": ["Solar Panels", "Mounting Structure", "Inverters", "Transformer", "Battery",
                                  "Cabling", "Installation", "Civil Work", "GST (13.8%)", "Total Project Cost"],
                    "Cost (₹)": [panel_cost_new, structure_cost_new, inverter_cost_new, transformer_cost_new,
                                 battery_cost_new, cabling_cost_new, installation_cost_new, civil_work_cost_new,
                                 gst_new, total_project_cost_new]
                })
                cost_df["Cost (₹)"] = cost_df["Cost (₹)"].round(0)
                st.dataframe(cost_df, use_container_width=True, hide_index=True)

                fig_cost = px.pie(cost_df[cost_df["Component"] != "Total Project Cost"], values="Cost (₹)", names="Component",
                                   hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
                fig_cost.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", title="Cost Breakdown")
                st.plotly_chart(fig_cost, use_container_width=True)

                # ==================== DISPLAY: SUBSIDY ====================
                st.markdown("#### 🏛️ Government Subsidy")
                sub_c1, sub_c2 = st.columns(2)
                sub_c1.markdown(f'<div class="animated-card"><div class="metric-label">Eligible Subsidy</div><div class="metric-value" style="color:#34d399;">₹{subsidy_new:,.0f}</div></div>', unsafe_allow_html=True)
                sub_c2.markdown(f'<div class="animated-card"><div class="metric-label">Final Cost After Subsidy</div><div class="metric-value" style="color:#fbbf24;">₹{final_cost_new:,.0f}</div></div>', unsafe_allow_html=True)
                st.info(subsidy_note)

                # ==================== DISPLAY: GENERATION ====================
                st.markdown("#### ⚡ Power Generation Estimate")
                g_c1, g_c2, g_c3, g_c4, g_c5 = st.columns(5)
                g_c1.metric("Hourly Gen", f"{hourly_gen_new} kW")
                g_c2.metric("Daily Gen", f"{daily_gen_new:,.0f} kWh")
                g_c3.metric("Monthly Gen", f"{monthly_gen_new:,.0f} kWh")
                g_c4.metric("Annual Gen", f"{annual_gen_new:,.0f} kWh")
                g_c5.metric("25-Yr Gen", f"{gen_25yr/1000:,.1f} MWh")

                # ==================== DISPLAY: FINANCIAL ANALYSIS ====================
                st.markdown("#### 📊 Financial Analysis")
                f_c1, f_c2, f_c3, f_c4 = st.columns(4)
                f_c1.markdown(f'<div class="animated-card"><div class="metric-label">Bill Before Solar</div><div class="metric-value">₹{bill_before_new:,.0f}</div></div>', unsafe_allow_html=True)
                f_c2.markdown(f'<div class="animated-card"><div class="metric-label">Bill After Solar</div><div class="metric-value">₹{bill_after_new:,.0f}</div></div>', unsafe_allow_html=True)
                f_c3.markdown(f'<div class="animated-card"><div class="metric-label">Monthly Saving</div><div class="metric-value" style="color:#34d399;">₹{monthly_saving_new:,.0f}</div></div>', unsafe_allow_html=True)
                f_c4.markdown(f'<div class="animated-card"><div class="metric-label">Annual Saving</div><div class="metric-value" style="color:#34d399;">₹{annual_saving_new:,.0f}</div></div>', unsafe_allow_html=True)

                f2_c1, f2_c2, f2_c3, f2_c4 = st.columns(4)
                f2_c1.markdown(f'<div class="animated-card"><div class="metric-label">Payback Period</div><div class="metric-value">{payback_years_new:.1f} Yrs</div></div>', unsafe_allow_html=True)
                f2_c2.markdown(f'<div class="animated-card"><div class="metric-label">ROI</div><div class="metric-value">{roi_pct_new:.1f}%</div></div>', unsafe_allow_html=True)
                f2_c3.markdown(f'<div class="animated-card"><div class="metric-label">25-Yr Profit</div><div class="metric-value">₹{net_lifetime_profit_new:,.0f}</div></div>', unsafe_allow_html=True)
                f2_c4.markdown(f'<div class="animated-card"><div class="metric-label">Maintenance (25 Yrs)</div><div class="metric-value">₹{maintenance_annual_new*25:,.0f}</div></div>', unsafe_allow_html=True)

                # ==================== DISPLAY: LAND ANALYSIS ====================
                st.markdown("#### 🗺️ Land Analysis")
                l_c1, l_c2, l_c3, l_c4, l_c5 = st.columns(5)
                l_c1.metric("Land Available", f"{land_available_sqft:,.0f} sq.ft")
                l_c2.metric("Land Required", f"{land_required_sqft:,.0f} sq.ft")
                l_c3.metric("Remaining Land", f"{remaining_land_sqft:,.0f} sq.ft")
                l_c4.metric("Max Possible Capacity", f"{max_possible_capacity:,.0f} kW")
                l_c5.metric("Land Utilization", f"{land_utilization_pct:.1f}%")
                st.progress(min(1.0, land_utilization_pct / 100))

                # ==================== MONTHLY SIMULATION ====================
                st.markdown("#### 📅 Monthly Simulation (Year 1)")
                months_new = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                seasonal_factors_new = [0.85, 0.95, 1.1, 1.2, 1.25, 1.15, 1.0, 0.95, 1.0, 1.1, 0.9, 0.8]

                monthly_sim_data = []
                for i, m in enumerate(months_new):
                    gen_m = daily_gen_new * 30 * seasonal_factors_new[i]
                    saved_m = min(gen_m, monthly_units_new) * tariff_new
                    revenue_m = saved_m
                    profit_m = revenue_m - (maintenance_annual_new / 12)
                    monthly_sim_data.append({
                        "Month": m,
                        "Solar Energy Generated (kWh)": round(gen_m, 1),
                        "Electricity Saved (₹)": round(saved_m, 0),
                        "Revenue (₹)": round(revenue_m, 0),
                        "Profit (₹)": round(profit_m, 0),
                        "Weather Factor": seasonal_factors_new[i]
                    })
                df_monthly_new = pd.DataFrame(monthly_sim_data)
                st.dataframe(df_monthly_new, use_container_width=True, hide_index=True)

                fig_sim = go.Figure()
                fig_sim.add_trace(go.Bar(x=df_monthly_new["Month"], y=df_monthly_new["Revenue (₹)"], name="Revenue", marker_color='#6366f1'))
                fig_sim.add_trace(go.Scatter(x=df_monthly_new["Month"], y=df_monthly_new["Profit (₹)"], name="Profit", mode='lines+markers', line=dict(color='#34d399', width=3)))
                fig_sim.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", title="Monthly Revenue vs Profit")
                st.plotly_chart(fig_sim, use_container_width=True)

                # ==================== YEARLY SIMULATION ====================
                st.markdown("#### 📈 Yearly Simulation")
                y_c1, y_c2, y_c3, y_c4 = st.columns(4)
                y_c1.markdown(f'<div class="animated-card"><div class="metric-label">Annual Generation</div><div class="metric-value">{annual_gen_new:,.0f} kWh</div></div>', unsafe_allow_html=True)
                y_c2.markdown(f'<div class="animated-card"><div class="metric-label">Annual Revenue</div><div class="metric-value">₹{annual_saving_new:,.0f}</div></div>', unsafe_allow_html=True)
                y_c3.markdown(f'<div class="animated-card"><div class="metric-label">Annual Profit</div><div class="metric-value">₹{annual_saving_new - maintenance_annual_new:,.0f}</div></div>', unsafe_allow_html=True)
                y_c4.markdown(f'<div class="animated-card"><div class="metric-label">ROI</div><div class="metric-value">{roi_pct_new:.1f}%</div></div>', unsafe_allow_html=True)

                # ==================== ENVIRONMENTAL IMPACT ====================
                st.markdown("#### 🌱 Environmental Impact")
                e_c1, e_c2, e_c3 = st.columns(3)
                e_c1.markdown(f'<div class="animated-card"><div style="font-size:30px;">🌍</div><div class="metric-value">{co2_saved_annual_new/1000:.1f} Tons</div><div class="metric-label">CO₂ Saved Annually</div></div>', unsafe_allow_html=True)
                e_c2.markdown(f'<div class="animated-card"><div style="font-size:30px;">🌳</div><div class="metric-value">{trees_equiv_new:.0f} Trees</div><div class="metric-label">Equivalent Trees Planted</div></div>', unsafe_allow_html=True)
                e_c3.markdown(f'<div class="animated-card"><div style="font-size:30px;">🔋</div><div class="metric-value">{annual_gen_new/1000:.1f} MWh</div><div class="metric-label">Green Energy Generated</div></div>', unsafe_allow_html=True)

                # ==================== COMMERCIAL PLANT RECOMMENDATION ====================
                st.markdown("#### 🧠 AI Commercial Plant Recommendation")
                rec_c1, rec_c2 = st.columns(2)
                with rec_c1:
                    st.markdown(f"""
                    **Best Panel Brand:** {panel_brand_new}

                    **Best Technology:** {panel_technology_new} — chosen for higher energy yield per sq.ft, which matters most given your {land_utilization_pct:.0f}% land utilization.

                    **Best Inverter:** {inverter_capacity_new}

                    **Battery Requirement:** {"Recommended, " + str(battery_capacity_kwh) + " kWh for backup/peak shaving" if battery_required_new else "Not required for a fully grid-connected setup"}
                    """)
                with rec_c2:
                    st.markdown(f"""
                    **Future Expansion:** {remaining_land_sqft:,.0f} sq.ft land still free — supports up to **{(remaining_land_sqft / sqft_per_kw_ground_mount):,.0f} kW** additional capacity later.

                    **Estimated Plant Life:** 25–30 years with standard panel degradation (~0.5%/year).

                    **Investment Quality:** {"Strong — payback under 5 years" if payback_years_new < 5 else "Good — payback in 5-8 years" if payback_years_new < 8 else "Moderate — consider phased rollout to improve ROI"}.

                    **Maintenance Schedule:** Panel cleaning every 15-20 days (higher frequency in dusty seasons), full inspection quarterly, inverter servicing annually.
                    """)

                # ==================== EXPORT CENTER ====================
                st.markdown("#### 📤 Export Center")
                exp_c1, exp_c2, exp_c3 = st.columns(3)

                with exp_c1:
                    csv_new = df_monthly_new.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV Report", data=csv_new,
                                        file_name=f"Solar_Plant_Simulation_{land_city}.csv", mime="text/csv",
                                        use_container_width=True, key="m3_csv_dl")

                with exp_c2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        cost_df.to_excel(writer, sheet_name='Cost Estimation', index=False)
                        df_monthly_new.to_excel(writer, sheet_name='Monthly Simulation', index=False)
                    st.download_button("📊 Download Excel Report", data=excel_buffer.getvalue(),
                                        file_name=f"Solar_Plant_Report_{land_city}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True, key="m3_excel_dl")

                with exp_c3:
                    def create_pdf_report_new():
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_fill_color(30, 41, 59)
                        pdf.rect(0, 0, 210, 40, 'F')
                        pdf.set_text_color(255, 255, 255)
                        pdf.set_font("Helvetica", 'B', 20)
                        pdf.cell(0, 20, 'Commercial/Industrial Solar Plant Report', 0, 1, 'C')
                        pdf.set_font("Helvetica", size=10)
                        pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                        pdf.set_text_color(0, 0, 0)
                        pdf.ln(10)

                        pdf.set_font("Helvetica", 'B', 14)
                        pdf.cell(0, 10, '1. Project Details', 0, 1)
                        pdf.set_font("Helvetica", size=11)
                        pdf.multi_cell(0, 6, f"Location: {land_city}, {land_state}\nPlant Type: {plant_type_new}\nLand Available: {land_available_sqft:,.0f} sq.ft\nBudget: Rs {budget_new:,.0f}")

                        pdf.ln(5)
                        pdf.set_font("Helvetica", 'B', 14)
                        pdf.cell(0, 10, '2. Recommended System', 0, 1)
                        pdf.set_font("Helvetica", size=11)
                        pdf.multi_cell(0, 6, f"Capacity: {final_capacity_new} kW\nPanels: {num_panels_new} x {panel_watt_new}W ({panel_technology_new})\nInverter: {inverter_capacity_new}\nLand Utilization: {land_utilization_pct:.1f}%")

                        pdf.ln(5)
                        pdf.set_font("Helvetica", 'B', 14)
                        pdf.cell(0, 10, '3. Financial Summary', 0, 1)
                        pdf.set_font("Helvetica", size=11)
                        pdf.multi_cell(0, 6, f"Total Project Cost: Rs {total_project_cost_new:,.0f}\nSubsidy: Rs {subsidy_new:,.0f}\nFinal Cost: Rs {final_cost_new:,.0f}\nPayback: {payback_years_new:.1f} Years\nROI: {roi_pct_new:.1f}%\nNet 25-Yr Profit: Rs {net_lifetime_profit_new:,.0f}")

                        return pdf.output(dest="S").encode("latin-1")

                    pdf_data_new = create_pdf_report_new()
                    st.download_button("📄 Download PDF Report", data=pdf_data_new,
                                        file_name=f"Solar_Plant_Report_{land_city}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                        mime="application/pdf", use_container_width=True, key="m3_pdf_dl")

with tab4:
    st.markdown('<div class="main-header"><h2>ℹ️ About AI Solar System</h2><p>Model Architecture & Tech Stack</p></div>', unsafe_allow_html=True)

    col_a1, col_a2, col_a3 = st.columns(3, gap="small")
    with col_a1:
        st.markdown(f'<div class="animated-card"><div style="font-size:22px;">🎯</div><div class="metric-value">{sco:.4f}</div><div class="metric-label">R² Score (Accuracy)</div></div>', unsafe_allow_html=True)
    with col_a2:
        st.markdown(f'<div class="animated-card"><div style="font-size:22px;">📐</div><div class="metric-value">{mae:.1f} kW</div><div class="metric-label">Mean Absolute Error</div></div>', unsafe_allow_html=True)
    with col_a3:
        st.markdown(f'<div class="animated-card"><div style="font-size:22px;">📉</div><div class="metric-value">{rmse:.1f} kW</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧠 How the Machine Learning Model Works")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""
        **1. Data Ingestion**
        Loads 50,000+ rows from `Solar_final.csv` spanning April-July 2020.

        **2. Feature Engineering**
        Extracts 16 core features including Cyclical Time Encoding (SIN/COS), Polynomial Interactions, and Lag Features to capture complex solar patterns.

        **3. Polynomial Transformation**
        Uses `PolynomialFeatures(degree=2)` to expand 16 features into ~150 interaction terms (e.g., Irradiation², Temp × Time).
        """)
    with col_e2:
        st.markdown("""
        **4. Model Training**
        A `DecisionTreeRegressor` is trained on the high-dimensional polynomial space to capture non-linear relationships without overfitting (due to tree logic).

        **5. Real-time Inference**
        When you input 5 parameters, the backend automatically generates the 150+ features and passes them to the model in milliseconds.

        **6. Financial Engine (New)**
        Separate logic engine calculates exact PM Surya Ghar subsidies, battery sizing, and 25-year amortization schedules.
        """)

    st.markdown("---")
    st.subheader("🛠️ Tech Stack")
    st.code("""
    Frontend  : Streamlit (Python), Custom CSS
    ML        : Scikit-learn (DecisionTreeRegressor, PolynomialFeatures)
    Data      : Pandas, NumPy
    Viz       : Plotly (Graph Objects, Express)
    Reports   : FPDF2 (PDF Generation)
    """, language='python')