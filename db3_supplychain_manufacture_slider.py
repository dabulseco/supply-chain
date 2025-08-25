# Dylan Bulseco, Institute for Future Intelligence, 2025 (v. 02)
# modify visual plot change with slider
# app_v2_live.py — CGT Supply Chain Simulator (Autologous with Substeps + Live Update, no button)
# -----------------------------------------------------------------------------------
# This revision removes the need to press a button: the sim runs automatically on each
# widget change and the plot persists. We:
#  - Always render a chart every rerun (no placeholders that get skipped)
#  - Use @st.cache_data keyed on a stable param hash to avoid unnecessary work
#  - Offer a "Pause live recompute" toggle
#  - Downsample in live mode for responsiveness; full fidelity when live mode is off
# -----------------------------------------------------------------------------------

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import simpy
import streamlit as st

# -------------------------
# Data structures
# -------------------------
@dataclass
class StepParams:
    mean: float  # days
    sd: float    # days

@dataclass
class ScenarioParams:
    n_patients: int
    interarrival_days: float
    # capacities
    apheresis_capacity: int
    mfg_rigs: int
    qc_stations: int
    clinic_slots_per_day: int
    # steps
    steps: Dict[str, StepParams]
    qc_days: float
    target_lead_time_days: float
    seed: int
    # vector availability model
    vector_delay_prob: float  # probability a given patient experiences vector delay
    vector_delay_mean: float  # days
    vector_delay_sd: float    # days

# -------------------------
# Helpers
# -------------------------

def truncated_gauss(mean: float, sd: float) -> float:
    if sd <= 0:
        return max(mean, 0.001)
    return max(random.gauss(mean, sd), 0.001)

# -------------------------
# Simulation core
# -------------------------

class CGTSim:
    def __init__(self, p: ScenarioParams):
        self.p = p
        self.env = simpy.Environment()
        # resources
        self.apheresis = simpy.Resource(self.env, capacity=p.apheresis_capacity)
        self.mfg = simpy.Resource(self.env, capacity=p.mfg_rigs)  # shared across substeps
        self.qc = simpy.Resource(self.env, capacity=p.qc_stations)
        # clinic daily slot booker
        self.clinic_slots = {}
        # event log
        self.events: List[Dict] = []

    def log(self, patient_id: str, step: str, start: float, end: float):
        self.events.append({
            'patient_id': patient_id,
            'step': step,
            'start_day': start,
            'end_day': end,
            'duration_days': end - start
        })

    def _do_step(self, res: simpy.Resource, duration_fn, pid: str, name: str):
        start = self.env.now
        with res.request() as req:
            yield req
            dur = duration_fn()
            yield self.env.timeout(dur)
        end = self.env.now
        self.log(pid, name, start, end)

    def _book_infusion_slot(self) -> float:
        day = math.floor(self.env.now)
        while True:
            if self.clinic_slots.get(day, 0) < self.p.clinic_slots_per_day:
                self.clinic_slots[day] = self.clinic_slots.get(day, 0) + 1
                return float(day)
            day += 1

    def patient_flow(self, idx: int):
        pid = f"P{idx:03d}"
        s = self.p.steps

        # Apheresis
        yield self.env.process(self._do_step(
            self.apheresis,
            lambda: truncated_gauss(s['Apheresis'].mean, s['Apheresis'].sd),
            pid, 'Apheresis'))

        # Ship-In
        start = self.env.now
        dur_si = truncated_gauss(s['Ship-In'].mean, s['Ship-In'].sd)
        yield self.env.timeout(dur_si)
        self.log(pid, 'Ship-In', start, self.env.now)

        # Activation (mfg rig)
        yield self.env.process(self._do_step(
            self.mfg,
            lambda: truncated_gauss(s['Activation'].mean, s['Activation'].sd),
            pid, 'Activation'))

        # Vector wait (pre-Transduction)
        if random.random() < self.p.vector_delay_prob:
            start = self.env.now
            vd = truncated_gauss(self.p.vector_delay_mean, self.p.vector_delay_sd)
            yield self.env.timeout(vd)
            self.log(pid, 'Vector Wait', start, self.env.now)

        # Transduction (mfg rig)
        yield self.env.process(self._do_step(
            self.mfg,
            lambda: truncated_gauss(s['Transduction'].mean, s['Transduction'].sd),
            pid, 'Transduction'))

        # Expansion (mfg rig)
        yield self.env.process(self._do_step(
            self.mfg,
            lambda: truncated_gauss(s['Expansion'].mean, s['Expansion'].sd),
            pid, 'Expansion'))

        # Fill/Finish
        start = self.env.now
        dur_ff = truncated_gauss(s['Fill/Finish'].mean, s['Fill/Finish'].sd)
        yield self.env.timeout(dur_ff)
        self.log(pid, 'Fill/Finish', start, self.env.now)

        # QC
        yield self.env.process(self._do_step(
            self.qc,
            lambda: truncated_gauss(self.p.qc_days, s['QC'].sd),
            pid, 'QC'))

        # Release
        start = self.env.now
        dur_rel = truncated_gauss(s['Release'].mean, s['Release'].sd)
        yield self.env.timeout(dur_rel)
        self.log(pid, 'Release', start, self.env.now)

        # Ship-Out
        start = self.env.now
        dur_so = truncated_gauss(s['Ship-Out'].mean, s['Ship-Out'].sd)
        yield self.env.timeout(dur_so)
        self.log(pid, 'Ship-Out', start, self.env.now)

        # Infusion (book daily slot)
        booked = self._book_infusion_slot()
        if booked > self.env.now:
            yield self.env.timeout(booked - self.env.now)
        start = self.env.now
        dur_inf = truncated_gauss(s['Infusion'].mean, s['Infusion'].sd)
        yield self.env.timeout(dur_inf)
        self.log(pid, 'Infusion', start, self.env.now)

    def run(self, horizon_days: float = 365.0) -> pd.DataFrame:
        random.seed(self.p.seed)
        np.random.seed(self.p.seed)

        def arrivals(env):
            for i in range(1, self.p.n_patients + 1):
                env.process(self.patient_flow(i))
                gap = max(np.random.exponential(self.p.interarrival_days), 0.01)
                yield env.timeout(gap)

        self.env.process(arrivals(self.env))
        self.env.run(until=horizon_days)
        return pd.DataFrame(self.events)

# -------------------------
# KPIs & plots
# -------------------------

def compute_kpis(df: pd.DataFrame, target_lead_time_days: float):
    if df.empty:
        return pd.DataFrame(), {
            'throughput': 0,
            'lead_time_median': float('nan'),
            'lead_time_p90': float('nan'),
            'on_time_pct': float('nan')
        }
    inf_end = df[df['step'] == 'Infusion'][['patient_id', 'end_day']].rename(columns={'end_day': 'inf_end'})
    aph_start = df[df['step'] == 'Apheresis'][['patient_id', 'start_day']].rename(columns={'start_day': 'aph_start'})
    lt = pd.merge(aph_start, inf_end, on='patient_id', how='inner')
    lt['lead_time'] = lt['inf_end'] - lt['aph_start']
    throughput = lt.shape[0]
    lead_time_median = float(lt['lead_time'].median()) if throughput else float('nan')
    lead_time_p90 = float(lt['lead_time'].quantile(0.9)) if throughput else float('nan')
    on_time_pct = float((lt['lead_time'] <= target_lead_time_days).mean() * 100) if throughput else float('nan')
    return lt, {
        'throughput': throughput,
        'lead_time_median': lead_time_median,
        'lead_time_p90': lead_time_p90,
        'on_time_pct': on_time_pct
    }


def make_gantt(df_events: pd.DataFrame, ref_date: pd.Timestamp):
    if df_events.empty:
        return None
    dfp = df_events.copy()
    dfp['Start'] = ref_date + pd.to_timedelta(dfp['start_day'], unit='D')
    dfp['Finish'] = ref_date + pd.to_timedelta(dfp['end_day'], unit='D')
    fig = px.timeline(dfp, x_start='Start', x_end='Finish', y='patient_id', color='step',
                      hover_data={'start_day': True, 'end_day': True, 'duration_days': ':.2f'})
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# -------------------------
# Streamlit UI (always-render pattern)
# -------------------------

st.set_page_config(page_title="CGT Simulator v2.1 — Live (no button)", layout="wide")
st.title("Cell & Gene Therapy Supply Chain Simulator — Autologous v2.1 (Live)")

# --- Live controls ---
with st.sidebar:
    st.header("Live Mode")
    live_mode = st.toggle("Recompute on change", value=True)
    pause_live = st.toggle("Pause live recompute", value=False)
    live_patients_cap = st.slider("Max patients in live mode", 5, 100, 30, step=5)
    live_horizon = st.slider("Max horizon in live mode (days)", 30, 365, 120, step=10)

with st.sidebar:
    st.header("Scenario Parameters")
    n_patients = st.slider("# Patients", 5, 200, 60, step=5)
    interarrival = st.slider("Mean Interarrival (days)", 0.05, 5.0, 0.5, step=0.05)

    st.subheader("Capacities")
    apheresis_cap = st.slider("Apheresis Chairs", 1, 20, 4)
    mfg_rigs = st.slider("Manufacturing Rigs", 1, 20, 4)
    qc_stations = st.slider("QC Stations", 1, 20, 3)
    clinic_slots = st.slider("Clinic Infusion Slots / day", 1, 40, 8)

    st.subheader("Durations (means ± sd) — days")
    aph_mean = st.number_input("Apheresis mean", 0.05, 5.0, 0.5, step=0.05)
    aph_sd = st.number_input("Apheresis sd", 0.0, 2.0, 0.1, step=0.05)

    shipin_mean = st.number_input("Ship-In mean", 0.05, 5.0, 0.75, step=0.05)
    shipin_sd = st.number_input("Ship-In sd", 0.0, 2.0, 0.15, step=0.05)

    st.markdown("**Manufacturing substeps**")
    act_mean = st.number_input("Activation mean", 0.05, 10.0, 1.0, step=0.05)
    act_sd = st.number_input("Activation sd", 0.0, 5.0, 0.2, step=0.05)

    trans_mean = st.number_input("Transduction mean", 0.05, 10.0, 0.5, step=0.05)
    trans_sd = st.number_input("Transduction sd", 0.0, 5.0, 0.1, step=0.05)

    exp_mean = st.number_input("Expansion mean", 0.1, 30.0, 5.5, step=0.1)
    exp_sd = st.number_input("Expansion sd", 0.0, 10.0, 1.0, step=0.1)

    ff_mean = st.number_input("Fill/Finish mean", 0.01, 5.0, 0.5, step=0.01)
    ff_sd = st.number_input("Fill/Finish sd", 0.0, 2.0, 0.1, step=0.01)

    qc_mode = st.selectbox("QC Method", ["USP <71> Sterility (14 d)", "Rapid Method (3 d)", "Custom"], index=0)
    custom_qc = st.number_input("Custom QC days", 0.5, 20.0, 5.0, step=0.5)

    rel_mean = st.number_input("Release mean", 0.01, 5.0, 0.25, step=0.01)
    rel_sd = st.number_input("Release sd", 0.0, 2.0, 0.05, step=0.01)

    shipout_mean = st.number_input("Ship-Out mean", 0.01, 5.0, 0.75, step=0.01)
    shipout_sd = st.number_input("Ship-Out sd", 0.0, 2.0, 0.15, step=0.01)

    inf_mean = st.number_input("Infusion mean", 0.01, 2.0, 0.2, step=0.01)
    inf_sd = st.number_input("Infusion sd", 0.0, 1.0, 0.05, step=0.01)

    st.subheader("Vector Availability (pre-Transduction)")
    vec_prob = st.slider("Delay probability", 0.0, 1.0, 0.2, step=0.05)
    vec_mean = st.number_input("Delay mean (days)", 0.0, 20.0, 2.0, step=0.1)
    vec_sd = st.number_input("Delay sd (days)", 0.0, 10.0, 0.5, step=0.1)

    st.subheader("Targets & Reproducibility")
    target_lt = st.slider("On-Time Lead Time Target (days)", 7, 180, 28)
    seed = st.number_input("Random Seed", 0, 10000, 42)

# QC duration resolve
if qc_mode.startswith("USP"):
    qc_days = 14.0
elif qc_mode.startswith("Rapid"):
    qc_days = 3.0
else:
    qc_days = float(custom_qc)

steps = {
    'Apheresis': StepParams(aph_mean, aph_sd),
    'Ship-In': StepParams(shipin_mean, shipin_sd),
    'Activation': StepParams(act_mean, act_sd),
    'Transduction': StepParams(trans_mean, trans_sd),
    'Expansion': StepParams(exp_mean, exp_sd),
    'Fill/Finish': StepParams(ff_mean, ff_sd),
    'QC': StepParams(qc_days, 0.5),
    'Release': StepParams(rel_mean, rel_sd),
    'Ship-Out': StepParams(shipout_mean, shipout_sd),
    'Infusion': StepParams(inf_mean, inf_sd)
}

full_params = ScenarioParams(
    n_patients=int(n_patients),
    interarrival_days=float(interarrival),
    apheresis_capacity=int(apheresis_cap),
    mfg_rigs=int(mfg_rigs),
    qc_stations=int(qc_stations),
    clinic_slots_per_day=int(clinic_slots),
    steps=steps,
    qc_days=float(qc_days),
    target_lead_time_days=float(target_lt),
    seed=int(seed),
    vector_delay_prob=float(vec_prob),
    vector_delay_mean=float(vec_mean),
    vector_delay_sd=float(vec_sd)
)

# Hash to key the cache
param_tuple = (
    full_params.n_patients, full_params.interarrival_days,
    full_params.apheresis_capacity, full_params.mfg_rigs, full_params.qc_stations, full_params.clinic_slots_per_day,
    steps['Apheresis'].mean, steps['Apheresis'].sd,
    steps['Ship-In'].mean, steps['Ship-In'].sd,
    steps['Activation'].mean, steps['Activation'].sd,
    steps['Transduction'].mean, steps['Transduction'].sd,
    steps['Expansion'].mean, steps['Expansion'].sd,
    steps['Fill/Finish'].mean, steps['Fill/Finish'].sd,
    full_params.qc_days,
    steps['Release'].mean, steps['Release'].sd,
    steps['Ship-Out'].mean, steps['Ship-Out'].sd,
    steps['Infusion'].mean, steps['Infusion'].sd,
    full_params.vector_delay_prob, full_params.vector_delay_mean, full_params.vector_delay_sd,
    full_params.target_lead_time_days, full_params.seed
)
param_hash = hash(param_tuple)

@st.cache_data(show_spinner=False)
def cached_run(hash_key: int, params: ScenarioParams, horizon: float):
    sim = CGTSim(params)
    df_events = sim.run(horizon_days=horizon)
    lt_df, kpis = compute_kpis(df_events, params.target_lead_time_days)
    return df_events, lt_df, kpis

# Determine effective params based on live vs paused
if live_mode and not pause_live:
    effective_params = ScenarioParams(
        n_patients=min(full_params.n_patients, int(live_patients_cap)),
        interarrival_days=full_params.interarrival_days,
        apheresis_capacity=full_params.apheresis_capacity,
        mfg_rigs=full_params.mfg_rigs,
        qc_stations=full_params.qc_stations,
        clinic_slots_per_day=full_params.clinic_slots_per_day,
        steps=full_params.steps,
        qc_days=full_params.qc_days,
        target_lead_time_days=full_params.target_lead_time_days,
        seed=full_params.seed,
        vector_delay_prob=full_params.vector_delay_prob,
        vector_delay_mean=full_params.vector_delay_mean,
        vector_delay_sd=full_params.vector_delay_sd
    )
    horizon = float(live_horizon)
else:
    effective_params = full_params
    horizon = 365.0

# Always compute (cached) and ALWAYS render
ref_date = pd.Timestamp('2025-01-01')
df_events, lt_df, kpis = cached_run(param_hash ^ int(horizon), effective_params, horizon)

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Throughput", f"{kpis['throughput']}")
c2.metric("Lead Time Median (d)", f"{kpis['lead_time_median']:.1f}" if not math.isnan(kpis['lead_time_median']) else "—")
c3.metric("Lead Time P90 (d)", f"{kpis['lead_time_p90']:.1f}" if not math.isnan(kpis['lead_time_p90']) else "—")
c4.metric("On‑Time Infusions (%)", f"{kpis['on_time_pct']:.0f}%" if not math.isnan(kpis['on_time_pct']) else "—")

# Chart (always present on page)
fig = make_gantt(df_events, ref_date)
if fig is not None:
    st.subheader("Patient Timelines (Gantt)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No events to display for current settings.")

# Tables & export
with st.expander("Details (Event Log & Lead Times)", expanded=False):
    st.write("Event Log")
    st.dataframe(df_events.sort_values(['patient_id','start_day']).reset_index(drop=True), use_container_width=True)
    st.write("Lead Times")
    st.dataframe(lt_df.sort_values('patient_id').reset_index(drop=True), use_container_width=True)

st.subheader("Export Results")
events_csv = df_events.to_csv(index=False).encode('utf-8')
lt_csv = lt_df.to_csv(index=False).encode('utf-8')
kpi_df = pd.DataFrame([kpis])
kpi_csv = kpi_df.to_csv(index=False).encode('utf-8')
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Download Event Log CSV", events_csv, file_name="events_live.csv", mime="text/csv")
with d2:
    st.download_button("Download Lead Times CSV", lt_csv, file_name="lead_times_live.csv", mime="text/csv")
with d3:
    st.download_button("Download KPI Summary CSV", kpi_csv, file_name="kpis_live.csv", mime="text/csv")

st.caption("Live mode down-samples for responsiveness; toggle off for full-fidelity numbers.")
