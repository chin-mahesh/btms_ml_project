import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

# Page Configuration
st.set_page_config(
    page_title="BTMS Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE ---
if 'entered_dashboard' not in st.session_state:
    st.session_state['entered_dashboard'] = False

if 'view_stm32_control' not in st.session_state:
    st.session_state['view_stm32_control'] = False

# Simulation State for STM32 Demo (Normal, Warning, Fault)
if 'stm32_sim_state' not in st.session_state:
    st.session_state['stm32_sim_state'] = 'NORMAL'

def enter_dashboard():
    st.session_state['entered_dashboard'] = True

def reset_to_landing():
    st.session_state['entered_dashboard'] = False
    st.session_state['view_stm32_control'] = False
    st.session_state['stm32_sim_state'] = 'NORMAL'

# Check query params for direct transition click
query_params = st.query_params
if "action" in query_params and query_params["action"] == "enter":
    st.session_state['entered_dashboard'] = True

# Helper to load external CSS file
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Helper to encode logo to base64
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_path = 'logo.png'
img_base64 = get_base64_image(logo_path)

# =========================================================
# COVER SCREEN
# =========================================================
if not st.session_state['entered_dashboard']:
    load_css('style.css')
    st.markdown("""
        <style>
        header[data-testid="stHeader"] {display: none !important;}
        .block-container { padding-top: 0rem !important; }
        </style>
    """, unsafe_allow_html=True)
    
    if img_base64 and os.path.exists('cover.html'):
        with open('cover.html', 'r') as f:
            html_content = f.read().replace("{img_base64}", img_base64)
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.error("Logo or cover.html file missing.")
        if st.button("Launch Dashboard", type="primary"):
            enter_dashboard()
            st.rerun()

# =========================================================
# MAIN ANALYTICS DASHBOARD OR FULL-SCREEN STM32 PAGE
# =========================================================
else:
    load_css('style.css')
    st.markdown("""
        <style>
        header[data-testid="stHeader"] {
            background-color: #05070a !important;
            background-image: radial-gradient(circle at top center, #111827 0%, #05070a 100%) !important;
            border-bottom: none !important;
            box-shadow: none !important;
        }
        .block-container {
            padding-top: 2rem !important;
        }
        div.stButton > button:not([kind="primary"]) {
            background-color: #05070a !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        div.stButton > button:not([kind="primary"]):hover {
            border-color: rgba(0, 210, 255, 0.4) !important;
            background-color: #111827 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- COMMON DATA LOADER ---
    @st.cache_data
    def load_data():
        csv_file = 'btms_prediction_results.csv'
        excel_file = 'btms_prediction_results.xlsx'
        if os.path.exists(csv_file):
            return pd.read_csv(csv_file)
        elif os.path.exists(excel_file):
            return pd.read_excel(excel_file, sheet_name='SOH_and_Anomaly_Predictions')
        return None

    df = load_data()

    if df is None:
        st.error("Prediction results missing! Run main.py first.")
        st.stop()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.markdown("### System Navigation")
    
    if st.session_state['view_stm32_control']:
        if st.sidebar.button("Return to Dashboard", type="primary"):
            st.session_state['view_stm32_control'] = False
            st.rerun()
            
        filtered_df = df.iloc[:1000].copy()
    else:
        if st.sidebar.button("Return to Cover"):
            st.query_params.clear()
            reset_to_landing()
            st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Activities")
        if st.sidebar.button("Live STM32 Telemetry & Control", type="primary"):
            st.session_state['view_stm32_control'] = True
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.header("Telemetry Controls")
        
        sample_size = st.sidebar.slider("Test Telemetry Range", min_value=100, max_value=len(df), value=1000, step=100)
        filtered_df = df.iloc[:sample_size].copy()

        st.sidebar.markdown("---")
        st.sidebar.subheader("Hardware & Cooling Architecture")
        st.sidebar.info("Edge Controller: STM32\n\nCooling: Active Pump + Radiator Fan + PCM Jacket")

    # =========================================================
    # FULL-SCREEN VIEW: LIVE STM32 TELEMETRY & CONTROL PAGE
    # =========================================================
    if st.session_state['view_stm32_control']:
        curr_state = st.session_state['stm32_sim_state']
        
        # Header title banner with dynamic state badge
        col_title, col_badge = st.columns([3, 1])
        with col_title:
            st.markdown('<p class="main-title">Battery Thermal Management System - Live Monitor</p>', unsafe_allow_html=True)
            st.markdown('<p class="sub-title">Pack Architecture: 8x 3.7V 18650 Li-Ion Cells </p>', unsafe_allow_html=True)
        
        with col_badge:
            if curr_state == 'NORMAL':
                st.markdown('<div style="background-color: #064e3b; color: #34d399; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; border: 1px solid #10b981;">STATE: NORMAL</div>', unsafe_allow_html=True)
            elif curr_state == 'WARNING':
                st.markdown('<div style="background-color: #78350f; color: #fbbf24; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; border: 1px solid #f59e0b;">STATE: THERMAL WARNING</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color: #7f1d1d; color: #fca5a5; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; border: 1px solid #ef4444;">STATE: PRE-RUNAWAY FAULT</div>', unsafe_allow_html=True)

        st.markdown("")
        
        # --- PHASE 1 PRESENTATION DEMO STATE CONTROLS ---
        st.markdown("### Controls:")
        dcol1, dcol2, dcol3 = st.columns(3)
        
        with dcol1:
            if st.button("Simulate Normal State", use_container_width=True):
                st.session_state['stm32_sim_state'] = 'NORMAL'
                st.rerun()
        with dcol2:
            if st.button("Simulate Thermal Warning", use_container_width=True):
                st.session_state['stm32_sim_state'] = 'WARNING'
                st.rerun()
        with dcol3:
            if st.button("Simulate Pre-Runaway Alert", use_container_width=True):
                st.session_state['stm32_sim_state'] = 'FAULT'
                st.rerun()

        st.markdown("---")
        
        # --- HARDWARE SENSOR TELEMETRY ARRAY (4x NTC & INA219) ---
        st.markdown("### Hardware Sensor Telemetry Array (NTC Probes & INA219)")
        
        if curr_state == 'NORMAL':
            ntc_temps = ["28.6 °C", "29.1 °C", "28.9 °C", "29.4 °C"]
            bus_voltage = "29.62 V"
            current_draw = "3.85 A"
            power_disp = "114.04 W"
        elif curr_state == 'WARNING':
            ntc_temps = ["51.0 °C", "54.2 °C", "52.8 °C", "53.5 °C"]
            bus_voltage = "28.90 V"
            current_draw = "8.40 A"
            power_disp = "242.76 W"
        else: # FAULT
            ntc_temps = ["77.4 °C", "82.1 °C", "79.8 °C", "81.5 °C"]
            bus_voltage = "27.45 V"
            current_draw = "14.20 A"
            power_disp = "389.79 W"

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("NTC Probe 1", ntc_temps[0], "Cell Group A")
        sc2.metric("NTC Probe 2", ntc_temps[1], "Cell Group B")
        sc3.metric("NTC Probe 3", ntc_temps[2], "Cell Group C")
        sc4.metric("NTC Probe 4", ntc_temps[3], "Cell Group D")
        sc5.metric("INA219 Sensor", bus_voltage, f"{current_draw} | {power_disp}")

        st.markdown("---")
        
        # --- DYNAMIC METRICS & AUTOMATED PUMP/FAN PWM BASED ON STATE ---
        if curr_state == 'NORMAL':
            pack_temp = "29.4 °C"
            pump_status = "Standby (0% PWM)"
            fan_status = "Standby (0% PWM)"
            pcm_state = "Solid Phase (Latent Heat Storage)"
            fault_reg = "0x00 (All Clear)"
            delta_val = "Optimal Range"
            target_pwm = 0
            target_fan = 0
        elif curr_state == 'WARNING':
            pack_temp = "52.8 °C"
            pump_status = "Active (60% PWM Loop)"
            fan_status = "Active (50% PWM Radiator)"
            pcm_state = "Solid-Liquid Transition (~50% Melt)"
            fault_reg = "0x02 (Warning Threshold Exceeded)"
            delta_val = "+12.8°C Spike"
            target_pwm = 60
            target_fan = 50
        else: # FAULT
            pack_temp = "78.2 °C"
            pump_status = "MAX OVERRIDE (100% PWM)"
            fan_status = "MAX OVERRIDE (100% PWM)"
            pcm_state = "Complete Liquid Phase (Max Capacity)"
            fault_reg = "0xFF (CRITICAL RUNAWAY RISK)"
            delta_val = "OVERHEAT FLUSH ACTIVE"
            target_pwm = 100
            target_fan = 100

        st.session_state['stm32_pwm_slider'] = target_pwm
        st.session_state['stm32_fan_slider'] = target_fan

        st.markdown("""
            <style>
            .metric-container {
                display: flex;
                gap: 15px;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #111827;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 16px;
                flex: 1;
            }
            .metric-label {
                color: #9ca3af;
                font-size: 0.85rem;
                margin-bottom: 6px;
            }
            .metric-value {
                color: #ffffff;
                font-size: 1.4rem;
                font-weight: 700;
                margin-bottom: 6px;
                word-wrap: break-word;
            }
            .metric-delta {
                color: #34d399;
                font-size: 0.8rem;
            }
            </style>
            
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">Pack Temp (NTC Array)</div>
                    <div class="metric-value">""" + pack_temp + """</div>
                    <div class="metric-delta">▲ """ + delta_val + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cooling Loop (Pump & Fan)</div>
                    <div class="metric-value">""" + pump_status + """</div>
                    <div class="metric-delta">▲ Fan: """ + fan_status + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Paraffin PCM Jacket</div>
                    <div class="metric-value">""" + pcm_state + """</div>
                    <div class="metric-delta">▲ Passive Buffer</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">STM32 Fault Register</div>
                    <div class="metric-value">""" + fault_reg + """</div>
                    <div class="metric-delta">▲ Edge Diagnostics</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            st.markdown("### Duty Cycle Overrides")
            pump_val = st.slider("Liquid Pump Speed (%)", min_value=0, max_value=100, step=5, key="stm32_pwm_slider")
            fan_val = st.slider("Radiator Fan Speed (%)", min_value=0, max_value=100, step=5, key="stm32_fan_slider")
            st.info(f"STM32 Telemetry: `VOLT:29.6V | PUMP_DUTY_CYCLE:{pump_val}% | FAN_DUTY_CYCLE:{fan_val}%`")
            
        with col_ctrl2:
            st.markdown("### Safety & Edge Interlocks")
            if curr_state == 'FAULT':
                st.error("CRITICAL WARNING: Thermal runaway probability > 85%. Emergency protocol armed.")
                if st.button("EMERGENCY SYSTEM KILLSWITCH (SHUTDOWN)", type="primary", key="killswitch_btn"):
                    st.error("POWER CONTACTOR TRIPPED! Entire system safely isolated and shut down by STM32 watchdog.")
            else:
                if st.button("Trigger Manual Thermal Flush", type="secondary", key="stm32_flush_btn"):
                    st.success("Manual cooling flush command (Pump/Fan 100%) transmitted to STM32 bus.")
            
            if st.button("Clear Fault Flags & Reset Registers", key="stm32_clear_btn"):
                st.session_state['stm32_sim_state'] = 'NORMAL'
                st.success("STM32 error registers successfully cleared.")
                st.rerun()

        st.markdown("### 8x 18650 Battery Pack Cell Heat Map & Telemetry Stream")
        
        if curr_state == 'NORMAL':
            cell_temps = [28.5, 29.1, 28.8, 29.4, 29.0, 28.7, 29.2, 28.9]
            cell_voltages = [3.72, 3.68, 3.71, 3.69, 3.70, 3.73, 3.67, 3.70]
        elif curr_state == 'WARNING':
            cell_temps = [51.2, 53.4, 52.0, 54.1, 52.8, 51.9, 53.5, 52.4]
            cell_voltages = [3.58, 3.55, 3.59, 3.54, 3.57, 3.60, 3.52, 3.56]
        else: # FAULT
            cell_temps = [76.5, 79.1, 77.4, 81.0, 78.5, 76.9, 80.2, 78.0]
            cell_voltages = [3.25, 3.12, 3.20, 3.05, 3.18, 3.22, 3.10, 3.15]
            
        cell_df = pd.DataFrame({
            "Cell ID": [f"Cell #{i+1} (3.7V 18650)" for i in range(8)],
            "Voltage (V)": cell_voltages,
            "Temperature (°C)": cell_temps,
            "Cooling Efficiency": ["Optimal" if curr_state=='NORMAL' else ("Active Flow" if curr_state=='WARNING' else "Max Dissipation")] * 8
        })
        st.dataframe(cell_df, use_container_width=True)

    # =========================================================
    # STANDARD ANALYTICS DASHBOARD VIEW (3 Tabs)
    # =========================================================
    else:
        st.markdown('<p class="main-title">Battery Thermal Management System - ML Analytics Hub</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">SOH Regression | Thermal Anomaly Tracking</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        avg_soh = filtered_df['Actual_SOH_%'].mean()
        avg_soh_pred = filtered_df['XGBoost_Predicted_SOH_%'].mean()
        total_anomalies = int(filtered_df['Predicted_Thermal_Anomaly'].sum())
        avg_error = filtered_df['SOH_Error_%'].mean()

        col1.metric("Avg Actual SOH", f"{avg_soh:.2f}%")
        col2.metric("Avg Predicted SOH", f"{avg_soh_pred:.2f}%")
        col3.metric("Avg SOH Error", f"{avg_error:.3f}%")
        col4.metric("Thermal Warning Events", f"{total_anomalies}", delta="Overheating Flag" if total_anomalies > 0 else "Normal", delta_color="inverse")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["SOH Tracking", "Thermal Anomalies", "Telemetry Data Logs"])

        with tab1:
            st.subheader("XGBoost State of Health (SOH) Estimation vs. Actual Ground Truth")
            fig_soh = px.scatter(
                filtered_df,
                x='Actual_SOH_%',
                y='XGBoost_Predicted_SOH_%',
                color='SOH_Error_%',
                color_continuous_scale='Viridis',
                labels={'Actual_SOH_%': 'Actual SOH (%)', 'XGBoost_Predicted_SOH_%': 'Predicted SOH (%)'}
            )
            fig_soh.add_shape(
                type='line', x0=filtered_df['Actual_SOH_%'].min(), y0=filtered_df['Actual_SOH_%'].min(),
                x1=filtered_df['Actual_SOH_%'].max(), y1=filtered_df['Actual_SOH_%'].max(),
                line=dict(color='Red', dash='dash', width=2)
            )
            fig_soh.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_soh, use_container_width=True)

        with tab2:
            st.subheader("Thermal Anomaly Probability & Warning Distribution")
            col_a, col_b = st.columns(2)
            with col_a:
                fig_prob = px.histogram(
                    filtered_df, 
                    x='Anomaly_Probability', 
                    nbins=30,
                    title="Classifier Thermal Overheating Probability Output Distribution",
                    color_discrete_sequence=['crimson']
                )
                fig_prob.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_prob, use_container_width=True)
                
            with col_b:
                anomaly_counts = filtered_df['Predicted_Thermal_Anomaly'].value_counts().reset_index()
                anomaly_counts.columns = ['Status', 'Count']
                anomaly_counts['Status'] = anomaly_counts['Status'].map({0: 'Normal Operating', 1: 'Thermal Warning Triggered'})
                
                fig_pie = px.pie(
                    anomaly_counts, 
                    names='Status', 
                    values='Count',
                    title="Operational Telemetry Health Split",
                    color='Status',
                    color_discrete_map={'Normal Operating': '#2ecc71', 'Thermal Warning Triggered': '#e74c3c'}
                )
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            st.subheader("Row-by-Row Inference Telemetry Log")
            st.dataframe(filtered_df, use_container_width=True)