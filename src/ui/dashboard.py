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
    "Live Simulator", "Experiments", "Research Analysis", "System Information"
])


def get_util_color_hex(pct: float) -> str:
    if pct < 50.0:
        return "#3fb950"
    if pct < 85.0:
        return "#d29922"
    return "#f85149"


# =========================================================
# TAB 1: LIVE SIMULATOR
# =========================================================
with tab_live:
    # A. Page Header
    st.header("Live Interactive Simulator — VM Load-Balancing Console")
    st.caption("Compare resource-aware Baseline selection with Random Forest under identical workload conditions.")
    
    # Compact System & Status Ribbon
    status_cols = st.columns([2, 2, 1, 1, 1])
    with status_cols[0]:
        if st.session_state.db_connected:
            st.markdown(f"**MySQL:** <span class='status-badge-ok'>Connected</span> `vm_selection_research`", unsafe_allow_html=True)
        else:
            st.markdown(f"**MySQL:** <span class='status-badge-error'>Disconnected</span> *(In-memory fallback)*", unsafe_allow_html=True)
    with status_cols[1]:
        if st.session_state.ml_selector:
            model_file = os.path.basename(st.session_state.ml_selector.model_path)
            st.markdown(f"**Model:** <span class='status-badge-ok'>Loaded</span> `{model_file}`", unsafe_allow_html=True)
        else:
            st.markdown(f"**Model:** <span class='status-badge-error'>Missing</span> *{st.session_state.get('ml_error', 'Unloaded')}*", unsafe_allow_html=True)
    with status_cols[2]:
        st.metric("Tick", st.session_state.current_tick)
    with status_cols[3]:
        st.metric("Provisioned VMs", len(st.session_state.env_baseline.vms))
    with status_cols[4]:
        st.metric("In Queue", len(st.session_state.queue))
        
    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 12px 0 16px 0;'>", unsafe_allow_html=True)

    # B. Algorithm Selection — At the Beginning of the Page
    st.markdown("### Select Algorithm / Execution Mode")
    exec_mode = st.radio(
        "Execution Mode",
        options=[
            "Compare Both — Baseline vs Random Forest",
            "Baseline Only",
            "Random Forest Only"
        ],
        index=0,
        label_visibility="collapsed"
    )
    if exec_mode == "Compare Both — Baseline vs Random Forest":
        st.caption("⚡ **Compare Both Active**: Each allocation evaluates the same workload against two independent, isolated simulation states (Baseline vs Random Forest) starting from equivalent conditions.")
    elif exec_mode == "Baseline Only":
        st.caption("⚙️ **Baseline Active**: Executes the deterministic weighted-utilization heuristic on the Baseline simulation state.")
    else:
        st.caption("🧠 **Random Forest Active**: Executes the trained ML regressor predicting multi-resource ground truth suitability.")

    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 16px 0 18px 0;'>", unsafe_allow_html=True)

    # C. Infrastructure State
    st.markdown("### Infrastructure State")
    
    if exec_mode == "Compare Both — Baseline vs Random Forest":
        view_env_choice = st.radio(
            "Display Pool",
            ["Baseline Environment State", "Random Forest Environment State"],
            horizontal=True,
            label_visibility="collapsed"
        )
        active_env = st.session_state.env_baseline if view_env_choice.startswith("Baseline") else st.session_state.env_rf
    elif exec_mode == "Baseline Only":
        active_env = st.session_state.env_baseline
    else:
        active_env = st.session_state.env_rf

    vms = list(active_env.vms.values())
    if not vms:
        st.info("No VMs provisioned yet. Create your first VM from VM Management below.")
    else:
        num_cards = len(vms)
        cols_per_row = 4 if num_cards >= 4 else (3 if num_cards == 3 else 2)
        grid_rows = [vms[i:i + cols_per_row] for i in range(0, len(vms), cols_per_row)]
        
        for row in grid_rows:
            card_cols = st.columns(cols_per_row)
            for idx, vm in enumerate(row):
                with card_cols[idx]:
                    c_pct = vm.cpu_utilization
                    m_pct = vm.memory_utilization
                    i_pct = vm.io_utilization
                    n_pct = vm.network_utilization
                    
                    status_badge = "<span class='status-badge-ok'>Active</span>"
                    if max(c_pct, m_pct, i_pct, n_pct) >= 90.0:
                        status_badge = "<span class='status-badge-error'>Overloaded</span>"
                    elif max(c_pct, m_pct, i_pct, n_pct) >= 80.0:
                        status_badge = "<span class='status-badge-warn'>Near-Cap</span>"
                        
                    card_html = f"""
                    <div class='infra-card'>
                        <div class='infra-card-header'>
                            <span>🖥️ <b>{vm.vm_id}</b></span>
                            <span>{status_badge}</span>
                        </div>
                        <div class='metric-row'>
                            <span class='metric-name'>CPU ({ResourceUnits.CPU}):</span>
                            <span class='metric-val'>{vm.cpu_alloc:,.0f} / {vm.cpu_cap:,.0f} ({c_pct:.1f}%)</span>
                        </div>
                        <div style='background-color:#21262d; border-radius:3px; height:6px; width:100%; margin-bottom:8px;'>
                            <div style='background-color:{get_util_color_hex(c_pct)}; width:{c_pct}%; height:6px; border-radius:3px;'></div>
                        </div>
                        
                        <div class='metric-row'>
                            <span class='metric-name'>RAM ({ResourceUnits.RAM}):</span>
                            <span class='metric-val'>{vm.mem_alloc:.1f} / {vm.mem_cap:.1f} ({m_pct:.1f}%)</span>
                        </div>
                        <div style='background-color:#21262d; border-radius:3px; height:6px; width:100%; margin-bottom:8px;'>
                            <div style='background-color:{get_util_color_hex(m_pct)}; width:{m_pct}%; height:6px; border-radius:3px;'></div>
                        </div>
                        
                        <div class='metric-row'>
                            <span class='metric-name'>Disk I/O ({ResourceUnits.DISK_IO}):</span>
                            <span class='metric-val'>{vm.io_alloc:,.0f} / {vm.io_cap:,.0f} ({i_pct:.1f}%)</span>
                        </div>
                        <div style='background-color:#21262d; border-radius:3px; height:6px; width:100%; margin-bottom:8px;'>
                            <div style='background-color:{get_util_color_hex(i_pct)}; width:{i_pct}%; height:6px; border-radius:3px;'></div>
                        </div>
                        
                        <div class='metric-row'>
                            <span class='metric-name'>Network ({ResourceUnits.NET}):</span>
                            <span class='metric-val'>{vm.net_alloc:,.0f} / {vm.net_cap:,.0f} ({n_pct:.1f}%)</span>
                        </div>
                        <div style='background-color:#21262d; border-radius:3px; height:6px; width:100%; margin-bottom:10px;'>
                            <div style='background-color:{get_util_color_hex(n_pct)}; width:{n_pct}%; height:6px; border-radius:3px;'></div>
                        </div>
                        
                        <div style='border-top:1px solid #21262d; padding-top:6px; display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e;'>
                            <span>Active Workloads:</span>
                            <span style='color:#f0f6fc; font-weight:600;'>{vm.active_workloads}</span>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 18px 0 20px 0;'>", unsafe_allow_html=True)

    # D. Live Comparison Workspace
    st.markdown("### Live Allocation Comparison")
    
    last_comp = st.session_state.last_comparison
    if not last_comp:
        st.info("No workload allocation has been executed yet. Click **Step (Allocate Next)** in Live Execution Controls below to trigger a live comparison.")
    else:
        wl = last_comp["workload"]
        st.markdown(f"""
        <div class='diff-callout'>
            <b>Evaluated Workload:</b> <code>{wl.workload_id}</code> ({wl.workload_type}) | 
            <b>Demand:</b> {ResourceUnits.format_cpu(wl.cpu_req)}, {ResourceUnits.format_ram(wl.mem_req)}, {ResourceUnits.format_io(wl.io_req)}, {ResourceUnits.format_net(wl.net_req)} | 
            <b>Duration:</b> {ResourceUnits.format_duration(wl.duration)}<br>
            <b>Outcome Analysis:</b> {last_comp.get('match_summary', 'Single algorithm execution recorded.')}
        </div>
        """, unsafe_allow_html=True)
        
        comp_cols = st.columns(2)
        with comp_cols[0]:
            st.markdown("#### Panel 1 — Baseline Scheduler")
            base_data = last_comp.get("baseline")
            if not base_data:
                st.caption("Baseline not evaluated in single-algorithm mode.")
            else:
                b_vm = base_data["selected_vm"]
                b_succ = base_data["success"]
                status_html = "<span class='status-badge-ok'>Allocated</span>" if b_succ else "<span class='status-badge-error'>Rejected</span>"
                
                panel_html = f"""
                <div class='infra-card'>
                    <div class='infra-card-header'>
                        <span>⚙️ <b>Baseline Heuristic</b> (Rule-Based Least Loaded)</span>
                        <span>{status_html}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Selected Target VM:</span>
                        <span class='metric-val' style='font-size:1.1rem; color:#58a6ff;'><b>{b_vm or 'None (Infeasible)'}</b></span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Feasible Candidates:</span>
                        <span class='metric-val'>{base_data['feasible_count']} VMs capable</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Measured Decision Time:</span>
                        <span class='metric-val' style='color:#3fb950;'><b>{base_data['decision_time_ms']:.4f} ms</b> ({base_data['decision_time_sec']:.6f} s)</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Selection Rationale:</span>
                        <span class='metric-val' style='font-family:sans-serif;'>{base_data['reason']}</span>
                    </div>
                </div>
                """
                st.markdown(panel_html, unsafe_allow_html=True)

        with comp_cols[1]:
            st.markdown("#### Panel 2 — Random Forest Selector")
            rf_data = last_comp.get("rf")
            if not rf_data:
                st.caption("Random Forest not evaluated in single-algorithm mode.")
            elif rf_data.get("error"):
                st.error(f"Random Forest Error: {rf_data['reason']}")
            else:
                r_vm = rf_data["selected_vm"]
                r_succ = rf_data["success"]
                status_html = "<span class='status-badge-ok'>Allocated</span>" if r_succ else "<span class='status-badge-error'>Rejected</span>"
                
                panel_html = f"""
                <div class='infra-card'>
                    <div class='infra-card-header'>
                        <span>🧠 <b>Random Forest Regressor</b> (Trained Suitability)</span>
                        <span>{status_html}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Selected Target VM:</span>
                        <span class='metric-val' style='font-size:1.1rem; color:#bc8cff;'><b>{r_vm or 'None (Infeasible)'}</b></span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Feasible Candidates:</span>
                        <span class='metric-val'>{rf_data['feasible_count']} VMs capable</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Measured Decision Time:</span>
                        <span class='metric-val' style='color:#d29922;'><b>{rf_data['decision_time_ms']:.4f} ms</b> ({rf_data['decision_time_sec']:.6f} s)</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-name'>Selection Rationale:</span>
                        <span class='metric-val' style='font-family:sans-serif;'>{rf_data['reason']}</span>
                    </div>
                </div>
                """
                st.markdown(panel_html, unsafe_allow_html=True)

    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 20px 0 22px 0;'>", unsafe_allow_html=True)

    # E. VM Management
    st.markdown("### VM Management")
    col_vm_create, col_vm_list = st.columns([1, 2])
    
    with col_vm_create:
        with st.form("vm_create_form"):
            st.markdown("**Provision New VM**")
            # Text input 0
            vm_id_input = st.text_input("VM Name", value=f"Node_{len(st.session_state.env.vms)+1}")
            # Number inputs 0..3 (strictly kept as first 4 number inputs in file)
            vm_cpu_input = st.number_input(f"CPU Capacity ({ResourceUnits.CPU})", min_value=1.0, value=1000.0, step=100.0)
            vm_mem_input = st.number_input(f"RAM Capacity ({ResourceUnits.RAM})", min_value=1.0, value=16.0, step=2.0)
            vm_io_input = st.number_input(f"Disk I/O Capacity ({ResourceUnits.DISK_IO})", min_value=1.0, value=5000.0, step=500.0)
            vm_net_input = st.number_input(f"Network Capacity ({ResourceUnits.NET})", min_value=1.0, value=1000.0, step=100.0)
            
            btn_prov_vm = st.form_submit_button("Provision VM", use_container_width=True)
            if btn_prov_vm:
                new_vm = VM(vm_id_input.strip(), vm_cpu_input, vm_mem_input, vm_io_input, vm_net_input)
                ok, msg = sync_add_vm(
                    new_vm,
                    st.session_state.env_baseline,
                    st.session_state.env_rf,
                    st.session_state.saved_vms,
                    st.session_state.live_exp_id
                )
                if ok:
                    st.session_state.history.append(f"Provisioned new VM: {new_vm.vm_id}")
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
                    
    with col_vm_list:
        st.markdown("**Current VM Pool**")
        vms_table_data = []
        for v in st.session_state.env_baseline.vms.values():
            base_v = st.session_state.env_baseline.vms.get(v.vm_id)
            rf_v = st.session_state.env_rf.vms.get(v.vm_id)
            vms_table_data.append({
                "VM Identifier": v.vm_id,
                f"CPU ({ResourceUnits.CPU})": f"{v.cpu_cap:,.0f}",
                f"RAM ({ResourceUnits.RAM})": f"{v.mem_cap:,.1f}",
                f"Disk I/O ({ResourceUnits.DISK_IO})": f"{v.io_cap:,.0f}",
                f"Net ({ResourceUnits.NET})": f"{v.net_cap:,.0f}",
                "Baseline Load": f"{base_v.active_workloads if base_v else 0} WLs",
                "RF Load": f"{rf_v.active_workloads if rf_v else 0} WLs",
            })
        if vms_table_data:
            st.dataframe(pd.DataFrame(vms_table_data), use_container_width=True, hide_index=True)
        else:
            st.caption("No VMs provisioned yet.")

    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 20px 0 22px 0;'>", unsafe_allow_html=True)

    # F. Workload Management and Queue
    st.markdown("### Workload Management")
    col_wl_create, col_wl_inventory = st.columns([1, 2])
    
    with col_wl_create:
        with st.form("wl_create_form"):
            st.markdown("**Create Workload**")
            # Text input 1
            wl_type_input = st.selectbox("Type", ["web_server", "database", "batch_job", "cache", "balanced"])
            # Number inputs 4..8 (strictly 5th through 9th number inputs in the app)
            wl_cpu_input = st.number_input(f"CPU Req ({ResourceUnits.CPU})", min_value=1.0, value=100.0, step=25.0)
            wl_mem_input = st.number_input(f"RAM Req ({ResourceUnits.RAM})", min_value=1.0, value=2.0, step=1.0)
            wl_io_input = st.number_input(f"I/O Req ({ResourceUnits.DISK_IO})", min_value=1.0, value=500.0, step=50.0)
            wl_net_input = st.number_input(f"Network Req ({ResourceUnits.NET})", min_value=1.0, value=100.0, step=25.0)
            wl_dur_input = st.number_input(f"Duration ({ResourceUnits.DURATION})", min_value=1, value=10, step=1)
            
            btn_queue_wl = st.form_submit_button("Queue Workload", use_container_width=True)
            if btn_queue_wl:
                new_id = f"WL_{len(st.session_state.queue) + len(st.session_state.env.allocations) + 1}"
                new_wl = Workload(new_id, wl_type_input, wl_cpu_input, wl_mem_input, wl_io_input, wl_net_input, 0, int(wl_dur_input))
                sync_add_workload(new_wl, st.session_state.saved_workloads, st.session_state.live_exp_id)
                st.session_state.queue.append(new_wl)
                st.session_state.history.append(f"Queued Workload: {new_wl.workload_id} ({wl_type_input})")
                st.success(f"Workload {new_id} queued.")
                st.rerun()

    with col_wl_inventory:
        st.markdown(f"**Workload Inventory ({len(st.session_state.saved_workloads)} Saved in MySQL)**")
        wl_table_data = []
        for w in st.session_state.saved_workloads.values():
            wl_table_data.append({
                "Workload ID": w.workload_id,
                "Profile": w.workload_type,
                f"CPU ({ResourceUnits.CPU})": f"{w.cpu_req:,.0f}",
                f"RAM ({ResourceUnits.RAM})": f"{w.mem_req:.1f}",
                f"Disk I/O ({ResourceUnits.DISK_IO})": f"{w.io_req:,.0f}",
                f"Net ({ResourceUnits.NET})": f"{w.net_req:,.0f}",
                "Duration": f"{w.duration} ticks"
            })
        if wl_table_data:
            st.dataframe(pd.DataFrame(wl_table_data), use_container_width=True, hide_index=True)
        else:
            st.caption("No workloads saved yet in inventory.")
            
        c_seed, c_q_sel = st.columns(2)
        with c_seed:
            if st.button("Load 10 Standard Research Workloads", use_container_width=True):
                added = seed_standard_workloads(st.session_state.saved_workloads, st.session_state.live_exp_id)
                st.success(f"Added {added} standard research workloads to MySQL inventory.")
                st.rerun()
        with c_q_sel:
            if st.session_state.saved_workloads:
                sel_wl = st.selectbox("Quick Enqueue Workload", list(st.session_state.saved_workloads.keys()), key="quick_q")
                if st.button("Enqueue Selected Workload", use_container_width=True):
                    wl_to_q = st.session_state.saved_workloads[sel_wl]
                    run_id = f"{wl_to_q.workload_id}_{int(time.time()) % 1000}"
                    q_copy = Workload(run_id, wl_to_q.workload_type, wl_to_q.cpu_req, wl_to_q.mem_req, wl_to_q.io_req, wl_to_q.net_req, 0, wl_to_q.duration)
                    st.session_state.queue.append(q_copy)
                    st.session_state.history.append(f"Queued Workload: {q_copy.workload_id} ({q_copy.workload_type})")
                    st.success(f"Enqueued {q_copy.workload_id}.")
                    st.rerun()

    # Advanced VM & Workload Modification Expander (inputs come after input 8)
    with st.expander("Advanced: Edit Capacities or Remove Entities"):
        if st.session_state.env_baseline.vms:
            st.markdown("**Modify Existing VM**")
            target_vm_id = st.selectbox("Select VM to Modify", list(st.session_state.env_baseline.vms.keys()), key="adv_vm_sel")
            vm_obj = st.session_state.env_baseline.vms[target_vm_id]
            c_edit1, c_edit2 = st.columns(2)
            with c_edit1:
                new_c = st.number_input(f"New CPU ({ResourceUnits.CPU})", min_value=1.0, value=float(vm_obj.cpu_cap), step=100.0, key="adv_edit_c")
                new_m = st.number_input(f"New RAM ({ResourceUnits.RAM})", min_value=1.0, value=float(vm_obj.mem_cap), step=2.0, key="adv_edit_m")
            with c_edit2:
                new_i = st.number_input(f"New Disk I/O ({ResourceUnits.DISK_IO})", min_value=1.0, value=float(vm_obj.io_cap), step=500.0, key="adv_edit_i")
                new_n = st.number_input(f"New Net ({ResourceUnits.NET})", min_value=1.0, value=float(vm_obj.net_cap), step=100.0, key="adv_edit_n")
                
            col_upd, col_del = st.columns(2)
            if col_upd.button("Apply Capacity Update", use_container_width=True):
                ok, msg = sync_update_vm(
                    target_vm_id, new_c, new_m, new_i, new_n,
                    st.session_state.env_baseline, st.session_state.env_rf,
                    st.session_state.saved_vms, st.session_state.live_exp_id
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
                    
            if col_del.button("Remove VM", type="secondary", use_container_width=True):
                ok, msg = sync_delete_vm(
                    target_vm_id, st.session_state.env_baseline, st.session_state.env_rf,
                    st.session_state.saved_vms, st.session_state.live_exp_id
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 20px 0 22px 0;'>", unsafe_allow_html=True)

    # G. Live Execution Controls
    st.markdown("### Live Execution Controls")
    
    ctl_col1, ctl_col2, ctl_col3, ctl_col4, ctl_col5 = st.columns(5)
    
    with ctl_col1:
        btn_step = st.button("Step (Allocate Next)", type="primary", use_container_width=True)
    with ctl_col2:
        btn_run_all = st.button("Run All Queued Workloads", use_container_width=True)
    with ctl_col3:
        btn_tick = st.button("Step Tick (+1 Tick)", use_container_width=True)
    with ctl_col4:
        btn_release = st.button("Release All Workloads", use_container_width=True)
    with ctl_col5:
        btn_reset = st.button("Reset Simulator", use_container_width=True)
        
    # Execution Logic
    if btn_step:
        if not st.session_state.queue:
            st.warning("No workloads in queue. Enqueue workloads from Workload Management above.")
        elif not st.session_state.env_baseline.vms:
            st.warning("No VMs available in infrastructure.")
        else:
            wl = st.session_state.queue.pop(0)
            wl.arrival_time = st.session_state.current_tick
            
            comp_res = evaluate_fair_allocation(
                exec_mode,
                wl,
                st.session_state.env_baseline,
                st.session_state.env_rf,
                st.session_state.baseline_selector,
                st.session_state.ml_selector,
                st.session_state.live_exp_id,
                st.session_state.current_tick
            )
            st.session_state.last_comparison = comp_res
            
            if comp_res["baseline"]:
                if comp_res["baseline"]["success"]:
                    st.session_state.success_count_base += 1
                    st.session_state.active_allocations_base.append({
                        "workload_id": wl.workload_id,
                        "vm_id": comp_res["baseline"]["selected_vm"],
                        "expires_at": st.session_state.current_tick + wl.duration
                    })
                    st.session_state.history.append(f"[Baseline] Allocated {wl.workload_id} -> {comp_res['baseline']['selected_vm']}")
                else:
                    st.session_state.reject_count_base += 1
                    st.session_state.history.append(f"[Baseline] REJECTED {wl.workload_id} ({comp_res['baseline']['reason']})")

            if comp_res["rf"]:
                if comp_res["rf"]["success"]:
                    st.session_state.success_count_rf += 1
                    st.session_state.active_allocations_rf.append({
                        "workload_id": wl.workload_id,
                        "vm_id": comp_res["rf"]["selected_vm"],
                        "expires_at": st.session_state.current_tick + wl.duration
                    })
                    st.session_state.history.append(f"[Random Forest] Allocated {wl.workload_id} -> {comp_res['rf']['selected_vm']}")
                else:
                    st.session_state.reject_count_rf += 1
                    st.session_state.history.append(f"[Random Forest] REJECTED {wl.workload_id} ({comp_res['rf']['reason']})")

            st.rerun()

    if btn_run_all:
        if not st.session_state.queue:
            st.warning("No workloads in queue.")
        else:
            processed_count = 0
            while st.session_state.queue:
                wl = st.session_state.queue.pop(0)
                wl.arrival_time = st.session_state.current_tick
                comp_res = evaluate_fair_allocation(
                    exec_mode,
                    wl,
                    st.session_state.env_baseline,
                    st.session_state.env_rf,
                    st.session_state.baseline_selector,
                    st.session_state.ml_selector,
                    st.session_state.live_exp_id,
                    st.session_state.current_tick
                )
                st.session_state.last_comparison = comp_res
                if comp_res["baseline"]:
                    if comp_res["baseline"]["success"]:
                        st.session_state.success_count_base += 1
                        st.session_state.active_allocations_base.append({
                            "workload_id": wl.workload_id,
                            "vm_id": comp_res["baseline"]["selected_vm"],
                            "expires_at": st.session_state.current_tick + wl.duration
                        })
                    else:
                        st.session_state.reject_count_base += 1
                if comp_res["rf"]:
                    if comp_res["rf"]["success"]:
                        st.session_state.success_count_rf += 1
                        st.session_state.active_allocations_rf.append({
                            "workload_id": wl.workload_id,
                            "vm_id": comp_res["rf"]["selected_vm"],
                            "expires_at": st.session_state.current_tick + wl.duration
                        })
                    else:
                        st.session_state.reject_count_rf += 1
                processed_count += 1
            st.session_state.history.append(f"Batch completed: processed {processed_count} queued workloads.")
            st.rerun()

    if btn_tick:
        new_tick, release_logs = advance_simulation_tick(
            st.session_state.env_baseline,
            st.session_state.env_rf,
            st.session_state.active_allocations_base,
            st.session_state.active_allocations_rf,
            st.session_state.current_tick,
            st.session_state.live_exp_id
        )
        st.session_state.current_tick = new_tick
        for log in release_logs:
            st.session_state.history.append(log)
            st.session_state.completed_count += 1
        st.rerun()

    if btn_release:
        for w_id in list(st.session_state.env_baseline.allocations.keys()):
            st.session_state.env_baseline.release_workload(w_id)
        for w_id in list(st.session_state.env_rf.allocations.keys()):
            st.session_state.env_rf.release_workload(w_id)
        st.session_state.active_allocations_base.clear()
        st.session_state.active_allocations_rf.clear()
        st.session_state.history.append("Released all active workloads from both simulation environments.")
        st.rerun()

    if btn_reset:
        reset_simulation(
            st.session_state.env_baseline,
            st.session_state.env_rf,
            st.session_state.live_exp_id
        )
        st.session_state.queue = []
        st.session_state.history = ["Simulation Reset: Live states returned to 0; VM/workload inventory preserved."]
        st.session_state.last_comparison = None
        st.session_state.current_tick = 0
        st.session_state.active_allocations_base.clear()
        st.session_state.active_allocations_rf.clear()
        st.rerun()

    # Live Execution Counters Metric Bar
    cnt_col1, cnt_col2, cnt_col3, cnt_col4 = st.columns(4)
    with cnt_col1:
        st.metric("Workloads Queued", len(st.session_state.queue))
    with cnt_col2:
        st.metric("Active Workloads (Base / RF)", f"{len(st.session_state.active_allocations_base)} / {len(st.session_state.active_allocations_rf)}")
    with cnt_col3:
        st.metric("Allocations (Base / RF)", f"{st.session_state.success_count_base} / {st.session_state.success_count_rf}")
    with cnt_col4:
        st.metric("Rejections (Base / RF)", f"{st.session_state.reject_count_base} / {st.session_state.reject_count_rf}")

    # Live Visualizations & Charts
    with st.expander("📊 Live Simulation Visualizations & Comparison Charts", expanded=False):
        vms = list(st.session_state.env_baseline.vms.values())
        if not vms:
            st.caption("No VMs provisioned yet. Create VMs above to enable live comparative visualizations.")
        else:
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.markdown("##### Resource Utilization by VM (Baseline vs Random Forest)")
                util_rows = []
                for v in vms:
                    b_v = st.session_state.env_baseline.vms.get(v.vm_id)
                    r_v = st.session_state.env_rf.vms.get(v.vm_id)
                    if b_v:
                        util_rows.append({"VM": v.vm_id, "Algorithm": "Baseline", "CPU %": b_v.cpu_utilization, "RAM %": b_v.memory_utilization, "Disk I/O %": b_v.io_utilization, "Net %": b_v.network_utilization})
                    if r_v:
                        util_rows.append({"VM": v.vm_id, "Algorithm": "Random Forest", "CPU %": r_v.cpu_utilization, "RAM %": r_v.memory_utilization, "Disk I/O %": r_v.io_utilization, "Net %": r_v.network_utilization})
                
                if util_rows:
                    df_u = pd.DataFrame(util_rows)
                    df_u_melt = pd.melt(df_u, id_vars=["VM", "Algorithm"], value_vars=["CPU %", "RAM %", "Disk I/O %", "Net %"], var_name="Resource", value_name="Utilization %")
                    fig_util = px.bar(
                        df_u_melt, x="VM", y="Utilization %", color="Algorithm", barmode="group",
                        facet_col="Resource",
                        color_discrete_map={"Baseline": "#58a6ff", "Random Forest": "#bc8cff"},
                        title="Per-VM Resource Utilization Comparison"
                    )
                    fig_util.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_util, use_container_width=True)

            with viz_col2:
                st.markdown("##### Decision Latency & Allocation Outcomes")
                c_sub1, c_sub2 = st.columns(2)
                with c_sub1:
                    last_comp = st.session_state.last_comparison
                    if last_comp and last_comp.get("baseline") and last_comp.get("rf"):
                        b_time = last_comp["baseline"]["decision_time_ms"]
                        r_time = last_comp["rf"]["decision_time_ms"]
                        df_lat = pd.DataFrame({
                            "Algorithm": ["Baseline", "Random Forest"],
                            "Decision Time (ms)": [b_time, r_time]
                        })
                        fig_lat = px.bar(
                            df_lat, x="Algorithm", y="Decision Time (ms)", color="Algorithm",
                            labels={"Decision Time (ms)": f"Decision Time ({ResourceUnits.DECISION_TIME})"},
                            color_discrete_map={"Baseline": "#58a6ff", "Random Forest": "#bc8cff"},
                            title="Latest Decision Latency"
                        )
                        fig_lat.update_layout(template="plotly_dark", height=280, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_lat, use_container_width=True)
                    else:
                        st.caption("No comparison step executed yet.")

                with c_sub2:
                    df_out = pd.DataFrame({
                        "Algorithm": ["Baseline", "Baseline", "Random Forest", "Random Forest"],
                        "Outcome": ["Allocated", "Rejected", "Allocated", "Rejected"],
                        "Count": [
                            st.session_state.success_count_base,
                            st.session_state.reject_count_base,
                            st.session_state.success_count_rf,
                            st.session_state.reject_count_rf
                        ]
                    })
                    fig_out = px.bar(
                        df_out, x="Algorithm", y="Count", color="Outcome", barmode="stack",
                        color_discrete_map={"Allocated": "#3fb950", "Rejected": "#f85149"},
                        title="Allocation Outcomes"
                    )
                    fig_out.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_out, use_container_width=True)

    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 20px 0 22px 0;'>", unsafe_allow_html=True)

    # H. Resource Events and Event History
    st.markdown("### Event History")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        evt_type_filter = st.selectbox("Filter by Event Type", ["All", "ALLOCATION_DECISION", "WORKLOAD_RELEASED", "VM_CREATED", "VM_UPDATED", "VM_DELETED", "WORKLOAD_CREATED", "SIMULATION_RESET"])
    with col_filter2:
        evt_alg_filter = st.selectbox("Filter by Algorithm", ["All", "BASELINE", "RANDOM_FOREST", "SYSTEM"])
        
    db_events = []
    if st.session_state.db_connected:
        db_events = SimulationEventRepository.get_recent_events(
            limit=25,
            experiment_id=st.session_state.live_exp_id,
            event_type=evt_type_filter,
            algorithm=evt_alg_filter
        )
        
    if db_events:
        evt_df = pd.DataFrame(db_events)[["created_at", "event_type", "algorithm", "workload_id", "vm_id", "result", "message"]]
        st.dataframe(evt_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("**Recent Session Activity Log:**")
        if st.session_state.history:
            for log in reversed(st.session_state.history[-8:]):
                st.markdown(f"- `{log}`")
        else:
            st.caption("No events recorded yet in this session.")

    st.markdown("<hr style='border:0; border-top:1px solid #21262d; margin: 20px 0 22px 0;'>", unsafe_allow_html=True)

    # I. MySQL Persistence Panel
    st.markdown("### MySQL Persistence & Database Transparency")
    
    db_status = get_cached_database_status()
    
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        if db_status["connected"]:
            st.markdown(f"""
            <div class='infra-card'>
                <div class='infra-card-header'>
                    <span>💾 <b>Connected to MySQL</b></span>
                    <span class='status-badge-ok'>LIVE</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-name'>Database:</span>
                    <span class='metric-val'>{db_status['database']}</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-name'>Endpoint:</span>
                    <span class='metric-val'>{db_status['host']}:{db_status['port']}</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-name'>Latest Record:</span>
                    <span class='metric-val'>{db_status.get('latest_timestamp', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"MySQL connection unavailable: {db_status.get('error', 'Unknown')}. Check .env credentials.")
            
        if st.button("Refresh Database Records", use_container_width=True):
            get_cached_database_status.clear()
            st.rerun()

    with p_col2:
        if db_status["connected"]:
            counts = db_status.get("counts", {})
            cnt_cols = st.columns(4)
            cnt_cols[0].metric("VM Records", counts.get("vms", 0))
            cnt_cols[1].metric("Workload Records", counts.get("workloads", 0))
            cnt_cols[2].metric("Decisions Logged", counts.get("allocation_decisions", 0))
            cnt_cols[3].metric("Simulation Events", counts.get("simulation_events", 0))
            
            with st.expander("Teacher Demonstration: Inspect Genuine MySQL Table Rows"):
                table_to_inspect = st.selectbox("Select Database Table to Inspect", [
                    "vms", "workloads", "allocation_decisions", "simulation_events", "experiments"
                ])
                rows = SystemRepository.get_recent_records(table_to_inspect, limit=10)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No rows found in this table.")


# =========================================================
# TAB 2: REPRODUCIBLE EXPERIMENTS
# =========================================================
with tab_exp:
    st.header("Reproducible Experiments")
    st.markdown("Run automated controlled matrices decoupled from the live interactive session.")
    
    with st.form("experiment_runner"):
        vms_input = st.number_input("VM Count", min_value=10, max_value=5000, value=50, step=10)
        wl_input = st.number_input("Workload Count", min_value=10, max_value=5000, value=100, step=10)
        seed_input = st.number_input("Random Seed", value=42)
        submit = st.form_submit_button("Run Comparison Experiment")
        
    if submit:
        if st.session_state.ml_selector is None:
            st.error("No ML Model found. Please train a model first via `scripts/train_model.py`.")
        else:
            with st.spinner("Running reproducible comparison (Baseline vs ML). Please wait..."):
                try:
                    # Uses standardized pipeline_path interface fixed at root cause
                    model_path = getattr(st.session_state.ml_selector, "pipeline_path", st.session_state.ml_selector.model_path)
                    engine = ExperimentEngine(ml_model_path=model_path)
                    engine.run_comparison(num_vms=vms_input, num_workloads=wl_input, seed=seed_input)
                    st.success("Experiment completed and saved to MySQL! Review results in Research Analysis tab.")
                except Exception as e:
                    st.error(f"Experiment failed: {e}")


# =========================================================
# TAB 3: RESEARCH ANALYSIS
# =========================================================
with tab_analysis:
    st.header("Research Analysis")
    st.markdown("Query historical experiment metrics logged into MySQL.")
    
    try:
        experiments = ExperimentRepository.get_all_experiments()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        experiments = []
        
    if experiments:
        exp_options = {f"ID: {e['experiment_id']} - {e['name']} ({e['created_at']})": e['experiment_id'] for e in experiments}
        selected_exp_label = st.selectbox("Select Experiment to View", list(exp_options.keys()))
        selected_exp_id = exp_options[selected_exp_label]
        
        try:
            results_raw = ExperimentRepository.get_experiment_results(selected_exp_id)
        except Exception as e:
            st.error(f"Failed to fetch metrics: {e}")
            results_raw = []
            
        if results_raw:
            data = []
            for r in results_raw:
                method = "baseline" if r['metric_name'].startswith("baseline_") else "random_forest"
                metric = r['metric_name'].replace("baseline_", "").replace("random_forest_", "")
                data.append({"Method": method, "Metric": metric, "Value": r['metric_value']})
                
            df = pd.DataFrame(data)
            df_pivot = df.pivot(index="Metric", columns="Method", values="Value").reset_index()
            
            st.subheader("Raw Metrics")
            st.dataframe(df_pivot, use_container_width=True)
            
            # Decision Time vs Balances Chart
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown("#### Mean Decision Time Comparison")
                time_df = df[df["Metric"] == "mean_decision_time"]
                if not time_df.empty:
                    time_df = time_df.copy()
                    time_df["Value_ms"] = time_df["Value"] * 1000.0
                    fig_time = px.bar(
                        time_df, x="Method", y="Value_ms", color="Method",
                        labels={"Value_ms": f"Decision Time ({ResourceUnits.DECISION_TIME})", "Method": "Algorithm"},
                        title="Decision Time Trade-Off (ms)",
                        color_discrete_map={"baseline": "#58a6ff", "random_forest": "#bc8cff"}
                    )
                    fig_time.update_layout(template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig_time, use_container_width=True)
                    
            with chart_cols[1]:
                st.markdown("#### Multi-Resource Balance Standard Deviation")
                bal_df = df[df["Metric"].str.contains("balance_std")]
                if not bal_df.empty:
                    fig_bal = px.bar(
                        bal_df, x="Metric", y="Value", color="Method", barmode="group",
                        labels={"Value": "Standard Deviation (Lower = More Balanced)", "Metric": "Resource Dimension"},
                        title="Load Balancing Deviation",
                        color_discrete_map={"baseline": "#58a6ff", "random_forest": "#bc8cff"}
                    )
                    fig_bal.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_bal, use_container_width=True)

            # Scalability trends (optimized single-batch query)
            st.subheader("Scalability Trends")
            agg_df = load_scalability_aggregates()
            if not agg_df.empty and "config_vms" in agg_df.columns and "config_workloads" in agg_df.columns:
                vm_scale_df = agg_df[agg_df["name"].str.contains("VM_Scale")]
                if not vm_scale_df.empty:
                    fig_vm_time = go.Figure()
                    fig_vm_time.add_trace(go.Scatter(x=vm_scale_df["config_vms"], y=vm_scale_df["baseline_mean_decision_time_mean"] * 1000.0, mode='lines+markers', name="Baseline", line=dict(color="#58a6ff")))
                    if "random_forest_mean_decision_time_mean" in vm_scale_df:
                        fig_vm_time.add_trace(go.Scatter(x=vm_scale_df["config_vms"], y=vm_scale_df["random_forest_mean_decision_time_mean"] * 1000.0, mode='lines+markers', name="Random Forest", line=dict(color="#bc8cff")))
                    fig_vm_time.update_layout(template="plotly_dark", xaxis_title="Number of VMs", yaxis_title=f"Mean Decision Time ({ResourceUnits.DECISION_TIME})", title="Decision Time vs VM Pool Scale")
                    st.plotly_chart(fig_vm_time, use_container_width=True)
            else:
                st.info("No scalability aggregates found. Run `scripts/run_scalability.py` to generate scalability curves.")
        else:
            st.warning("No metrics found for this experiment.")
    else:
        st.info("No experiments found. Run an experiment to see data.")


# =========================================================
# TAB 4: SYSTEM INFORMATION
# =========================================================
with tab_sys:
    st.header("Host Machine Information")
    st.markdown("*(Used exclusively to run the Python simulation sandbox, not the simulated VM resources).*")
    
    col_sys1, col_sys2 = st.columns(2)
    with col_sys1:
        st.markdown(f"**Operating System:** {platform.system()} {platform.release()} ({platform.architecture()[0]})")
        st.markdown(f"**Processor:** {platform.processor()}")
        st.markdown(f"**Physical Cores:** {psutil.cpu_count(logical=False)}")
        st.markdown(f"**Logical Cores:** {psutil.cpu_count(logical=True)}")
    with col_sys2:
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        st.markdown(f"**Total Host RAM:** {total_ram:.2f} GB")
        st.markdown(f"**Python Version:** {platform.python_version()}")
        st.markdown(f"**Streamlit Version:** {st.__version__}")
        
    st.markdown("---")
    st.subheader("MySQL Server Telemetry")
    db_stat = get_cached_database_status()
    st.json(db_stat)
