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
from src.database.repositories import ExperimentRepository
from src.evaluation.experiment_engine import ExperimentEngine
from src.simulation.environment import SimulationEnvironment
from src.simulation.vm import VM
from src.simulation.workload import Workload
from src.algorithms.baseline import BaselineSelector
from src.algorithms.ml_selector import MLSelector

st.set_page_config(page_title="Intelligent VM Selection", layout="wide", initial_sidebar_state="collapsed")

# Custom Apple-inspired CSS
st.markdown("""
<style>
    /* Global colors and fonts */
    :root {
        --bg-color: #1c1c1e;
        --surface-color: #2c2c2e;
        --text-silver: #a1a1a6;
        --text-white: #f5f5f7;
        --accent-blue: #0a84ff;
        --border-color: #38383a;
        --status-green: #30d158;
        --status-amber: #ffd60a;
        --status-red: #ff453a;
    }
    
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-white);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-white) !important;
        font-weight: 500;
    }
    
    p, span, div {
        color: var(--text-silver);
    }
    
    .stButton>button {
        background-color: var(--surface-color);
        border: 1px solid var(--border-color);
        color: var(--text-white);
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: var(--accent-blue);
        border-color: var(--accent-blue);
        color: #ffffff;
    }
    
    /* Table styling for VMs */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 1rem;
    }
    .dataframe th {
        background-color: var(--surface-color);
        color: var(--text-white);
        font-weight: 500;
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid var(--border-color);
    }
    .dataframe td {
        background-color: var(--bg-color);
        color: var(--text-silver);
        padding: 8px;
        border-bottom: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# Initialization of Session State
if 'env' not in st.session_state:
    st.session_state.env = SimulationEnvironment()
if 'queue' not in st.session_state:
    st.session_state.queue = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'ml_selector' not in st.session_state:
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    joblibs = [f for f in os.listdir(models_dir) if f.endswith(".joblib")] if os.path.exists(models_dir) else []
    if joblibs:
        st.session_state.ml_selector = MLSelector(os.path.join(models_dir, sorted(joblibs)[-1]))
    else:
        st.session_state.ml_selector = None
if 'baseline_selector' not in st.session_state:
    st.session_state.baseline_selector = BaselineSelector()

# Top Navigation Tabs
tab_live, tab_exp, tab_analysis, tab_sys = st.tabs([
    "Live Simulator", "Experiments", "Research Analysis", "System Information"
])

def get_util_color(pct):
    if pct < 50: return "🟢"
    if pct < 85: return "🟡"
    return "🔴"

with tab_live:
    st.header("Live Interactive Simulator")
    
    col_sim1, col_sim2 = st.columns([2, 1])
    
    with col_sim1:
        st.subheader("Infrastructure State")
        vms = st.session_state.env.vms.values()
        if not vms:
            st.info("No VMs provisioned. Create a VM below to begin.")
        else:
            vm_data = []
            for vm in vms:
                c_pct = (vm.cpu_alloc / vm.cpu_cap) * 100
                m_pct = (vm.mem_alloc / vm.mem_cap) * 100
                i_pct = (vm.io_alloc / vm.io_cap) * 100
                n_pct = (vm.net_alloc / vm.net_cap) * 100
                
                vm_data.append({
                    "VM ID": vm.vm_id,
                    "CPU": f"{get_util_color(c_pct)} {c_pct:.1f}% ({vm.cpu_alloc}/{vm.cpu_cap})",
                    "RAM": f"{get_util_color(m_pct)} {m_pct:.1f}% ({vm.mem_alloc}/{vm.mem_cap})",
                    "I/O": f"{get_util_color(i_pct)} {i_pct:.1f}% ({vm.io_alloc}/{vm.io_cap})",
                    "Net": f"{get_util_color(n_pct)} {n_pct:.1f}% ({vm.net_alloc}/{vm.net_cap})",
                    "Workloads": vm.active_workloads
                })
            st.table(pd.DataFrame(vm_data))
            
        st.subheader("Event History")
        for log in reversed(st.session_state.history[-10:]):
            st.markdown(f"- {log}")
            
    with col_sim2:
        st.subheader("Control Panel")
        
        with st.expander("Create VM", expanded=not bool(vms)):
            with st.form("vm_create_form"):
                vm_id = st.text_input("VM Name", value=f"Node_{len(vms)+1}")
                vm_cpu = st.number_input("CPU Capacity", min_value=1.0, value=1000.0)
                vm_mem = st.number_input("RAM Capacity", min_value=1.0, value=16.0)
                vm_io = st.number_input("I/O Capacity", min_value=1.0, value=5000.0)
                vm_net = st.number_input("Network Capacity", min_value=1.0, value=1000.0)
                if st.form_submit_button("Provision VM"):
                    if vm_id in st.session_state.env.vms:
                        st.error("VM Name already exists!")
                    else:
                        st.session_state.env.add_vm(VM(vm_id, vm_cpu, vm_mem, vm_io, vm_net))
                        st.session_state.history.append(f"Provisioned new VM: {vm_id}")
                        st.rerun()
                        
        with st.expander("Create Workload"):
            with st.form("wl_create_form"):
                wl_type = st.selectbox("Type", ["web_server", "database", "batch_job", "cache"])
                wl_cpu = st.number_input("CPU Req", min_value=1.0, value=100.0)
                wl_mem = st.number_input("RAM Req", min_value=1.0, value=2.0)
                wl_io = st.number_input("I/O Req", min_value=1.0, value=500.0)
                wl_net = st.number_input("Network Req", min_value=1.0, value=100.0)
                wl_dur = st.number_input("Duration (ticks)", min_value=1, value=10)
                if st.form_submit_button("Queue Workload"):
                    wl = Workload(f"WL_{len(st.session_state.queue) + len(st.session_state.env.allocations) + 1}", 
                                  wl_type, wl_cpu, wl_mem, wl_io, wl_net, 0, wl_dur)
                    st.session_state.queue.append(wl)
                    st.session_state.history.append(f"Queued Workload: {wl.workload_id} ({wl_type})")
                    st.rerun()
                    
        st.markdown("---")
        alg = st.radio("Selector Algorithm", ["Baseline", "Random Forest"])
        
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Step (Allocate Next)", use_container_width=True):
            if not st.session_state.queue:
                st.warning("No workloads in queue.")
            elif not st.session_state.env.vms:
                st.warning("No VMs to allocate to.")
            else:
                wl = st.session_state.queue.pop(0)
                selector = st.session_state.baseline_selector if alg == "Baseline" else st.session_state.ml_selector
                
                if selector is None:
                    st.error("ML Selector model not loaded.")
                    st.session_state.queue.insert(0, wl)
                else:
                    feasible_vms = st.session_state.env.get_feasible_vms(wl)
                    if feasible_vms:
                        selected_vm, _ = selector.select_vm(feasible_vms, wl)
                        if selected_vm:
                            st.session_state.env.allocate_workload(selected_vm.vm_id, wl)
                            st.session_state.history.append(f"[{alg}] Allocated {wl.workload_id} -> {selected_vm.vm_id}")
                        else:
                            st.session_state.history.append(f"[{alg}] FAILED to allocate {wl.workload_id} (VMs infeasible)")
                    else:
                        st.session_state.history.append(f"[{alg}] FAILED to allocate {wl.workload_id} (No feasible VMs)")
                st.rerun()
                
        if col_btn2.button("Release All Workloads", use_container_width=True):
            allocated_ids = list(st.session_state.env.allocations.keys())
            for w_id in allocated_ids:
                st.session_state.env.release_workload(w_id)
            st.session_state.history.append(f"Released {len(allocated_ids)} workloads.")
            st.rerun()
            
        if st.button("Reset Simulator", type="primary", use_container_width=True):
            st.session_state.env = SimulationEnvironment()
            st.session_state.queue = []
            st.session_state.history = ["Simulation Reset."]
            st.rerun()

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
            st.error("No ML Model found. Please train a model first.")
        else:
            with st.spinner("Running reproducible comparison (Baseline vs ML). Please wait..."):
                try:
                    engine = ExperimentEngine(ml_model_path=st.session_state.ml_selector.pipeline_path)
                    engine.run_comparison(num_vms=vms_input, num_workloads=wl_input, seed=seed_input)
                    st.success("Experiment completed and saved! Check Research Analysis tab.")
                except Exception as e:
                    st.error(f"Experiment failed: {e}")

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
            
            # Scalability trends
            st.subheader("Scalability Trends")
            agg_exps = [e for e in experiments if e["name"].startswith("AGG_")]
            if agg_exps:
                agg_data = []
                for e in agg_exps:
                    r_raw = ExperimentRepository.get_experiment_results(e['experiment_id'])
                    row = {"name": e["name"]}
                    for r in r_raw:
                        row[r['metric_name']] = r['metric_value']
                    agg_data.append(row)
                
                agg_df = pd.DataFrame(agg_data)
                if not agg_df.empty and "config_vms" in agg_df.columns and "config_workloads" in agg_df.columns:
                    vm_scale_df = agg_df[agg_df["name"].str.contains("VM_Scale")]
                    if not vm_scale_df.empty:
                        st.markdown("Decision Time vs VM Count")
                        fig_vm_time = go.Figure()
                        fig_vm_time.add_trace(go.Scatter(x=vm_scale_df["config_vms"], y=vm_scale_df["baseline_mean_decision_time_mean"], mode='lines+markers', name="Baseline"))
                        if "random_forest_mean_decision_time_mean" in vm_scale_df:
                            fig_vm_time.add_trace(go.Scatter(x=vm_scale_df["config_vms"], y=vm_scale_df["random_forest_mean_decision_time_mean"], mode='lines+markers', name="Random Forest"))
                        fig_vm_time.update_layout(xaxis_title="Number of VMs", yaxis_title="Mean Decision Time (s)")
                        st.plotly_chart(fig_vm_time, use_container_width=True)
            else:
                st.info("No scalability aggregates found.")
        else:
            st.warning("No metrics found for this experiment.")
    else:
        st.info("No experiments found. Run an experiment to see data.")

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
