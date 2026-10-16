import os
import sys
import platform
import psutil
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.simulation.units import ResourceUnits
from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.simulation.environment import SimulationEnvironment
from src.algorithms.baseline import BaselineSelector
from src.algorithms.ml_selector import MLSelector
from src.database.connection import get_db_connection
from src.database.repositories import (
    ExperimentRepository,
    VMRepository,
    WorkloadRepository,
    AllocationRepository,
    SimulationEventRepository,
    SystemRepository
)
from src.evaluation.experiment_engine import ExperimentEngine
from src.ui.live_state import (
    initialize_persistent_state,
    load_ml_selector,
    seed_standard_workloads,
    sync_add_vm,
    sync_update_vm,
    sync_delete_vm,
    sync_add_workload,
    sync_update_workload,
    sync_delete_workload,
    evaluate_fair_allocation,
    advance_simulation_tick,
    reset_simulation
)

st.set_page_config(page_title="Multi-Resource VM Selection Console", layout="wide", initial_sidebar_state="collapsed")

# Custom Dark Infrastructure Console CSS
st.markdown("""
<style>
    :root {
        --bg-main: #0d1117;
        --bg-panel: #161b22;
        --bg-header: #1f242c;
        --border-main: #30363d;
        --border-subtle: #21262d;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-blue: #58a6ff;
        --accent-green: #3fb950;
        --accent-amber: #d29922;
        --accent-red: #f85149;
        --accent-purple: #bc8cff;
    }
    
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.2px;
    }
    
    p, span, label {
        color: var(--text-secondary);
    }
    
    .infra-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-main);
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 14px;
    }
    
    .infra-card-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 8px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .status-badge-ok {
        background-color: rgba(63, 185, 80, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-badge-warn {
        background-color: rgba(210, 153, 34, 0.15);
        color: var(--accent-amber);
        border: 1px solid rgba(210, 153, 34, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-badge-error {
        background-color: rgba(248, 81, 73, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(248, 81, 73, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    
    .metric-name {
        color: var(--text-secondary);
    }
    
    .metric-val {
        color: var(--text-primary);
        font-family: monospace;
        font-weight: 500;
    }
    
    .diff-callout {
        background-color: rgba(88, 166, 255, 0.08);
        border-left: 4px solid var(--accent-blue);
        padding: 10px 14px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 16px;
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .stButton>button {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-main);
        color: var(--text-primary);
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    
    .stButton>button:hover {
        background-color: #21262d;
        border-color: var(--accent-blue);
        color: var(--accent-blue);
    }
    
    .stTable, .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_ml_selector():
    return load_ml_selector()


@st.cache_data(ttl=15)
def get_cached_database_status():
    return SystemRepository.get_database_status()


@st.cache_data(ttl=30)
def load_scalability_aggregates():
    try:
        exps = ExperimentRepository.get_all_experiments()
        agg_exps = [e for e in exps if e["name"].startswith("AGG_")]
        if not agg_exps:
            return pd.DataFrame()
            
        ids = [e['experiment_id'] for e in agg_exps]
        format_strings = ','.join(['%s'] * len(ids))
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    f"SELECT experiment_id, metric_name, metric_value FROM experiment_results WHERE experiment_id IN ({format_strings})",
                    tuple(ids)
                )
                all_results = cursor.fetchall()
                
        exp_metrics = {}
        for r in all_results:
            exp_metrics.setdefault(r['experiment_id'], {})[r['metric_name']] = r['metric_value']
            
        agg_data = []
        for e in agg_exps:
            row = {"name": e["name"]}
            row.update(exp_metrics.get(e['experiment_id'], {}))
            agg_data.append(row)
        return pd.DataFrame(agg_data)
    except Exception as e:
        return pd.DataFrame()


# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if 'initialized' not in st.session_state:
    state_bundle = initialize_persistent_state()
    st.session_state.db_connected = state_bundle["db_connected"]
    st.session_state.live_exp_id = state_bundle["live_exp_id"]
    st.session_state.saved_vms = state_bundle["saved_vms"]
    st.session_state.saved_workloads = state_bundle["saved_workloads"]
    st.session_state.env_baseline = state_bundle["env_baseline"]
    st.session_state.env_rf = state_bundle["env_rf"]
    st.session_state.env = st.session_state.env_baseline
    st.session_state.queue = []
    st.session_state.history = []
    st.session_state.current_tick = 0
    st.session_state.active_allocations_base = []
    st.session_state.active_allocations_rf = []
    st.session_state.last_comparison = None
    st.session_state.success_count_base = 0
    st.session_state.reject_count_base = 0
    st.session_state.success_count_rf = 0
    st.session_state.reject_count_rf = 0
    st.session_state.completed_count = 0
    
    ml_sel, ml_err = get_cached_ml_selector()
    st.session_state.ml_selector = ml_sel
    st.session_state.ml_error = ml_err
    st.session_state.baseline_selector = BaselineSelector()
    st.session_state.initialized = True

# Top Navigation Tabs


tab_live, tab_exp, tab_analysis, tab_sys = st.tabs([
    "Live Simulator", "Reproducible Experiments", "Research Analysis", "System Information"
])
with tab_live:
    st.info("Live Simulator loading...")
