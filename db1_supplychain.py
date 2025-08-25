# app.py — CGT Supply Chain Simulator (Autologous v1)
# --------------------------------------------------
# Features:
# 1) Autologous (CAR‑T–style) "vein‑to‑vein" scheduler using discrete‑event simulation (SimPy)
# 2) QC toggle: USP <71> sterility (14 d) vs Rapid (e.g., 3 d) vs Custom
# 3) KPI cards (lead time median/P90, on‑time infusion %, throughput) + Gantt timeline
# 4) CSV export of simulation event log and KPI summary
# --------------------------------------------------
# D. Bulseco, Institute for Future Intelligence, 2025 (version 0.1)

import math
import random
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import simpy
import plotly.express as px
import streamlit as st

# -------------------------
# Utility data structures
# -------------------------
@dataclass
class StepParams:
    mean: float  # in days
    sd: float    # in days (Gaussian noise; truncated at 0)

@dataclass
class ScenarioParams:
    n_patients: int
    interarrival_days: float  # mean interarrival time between patient collections (days)
    apheresis_capacity: int   # concurrent capacity ("chairs")
    mfg_rigs: int             # concurrent autologous manufacturing capacity
    qc_stations: int          # concurrent QC capacity
    clinic_slots_per_day: int # discrete infusion slots available per day
    steps: Dict[str, StepParams]
    qc_days: float            # QC duration (overrides steps['QC'] mean)
    target_lead_time_days: float
    seed: int

# -------------------------
# Random duration helpers
# -------------------------

def truncated_gauss(mean: float, sd: float) -> float:
    """Gaussian truncated at 0.001 days to avoid negative/zero durations."""
    if sd <= 0:
        return max(mean, 0.001)
    x = random.gauss(mean, sd)
    return max(x, 0.001)

# -------------------------
# Simulation core (SimPy)
# -------------------------

class CGTSim:
    def __init__(self, params: ScenarioParams):
        self.p = params
        self.env = simpy.Environment()
        # Resources
        self.apheresis = simpy.Resource(self.env, capacity=self.p.apheresis_capacity)
        self.mfg = simpy.Resource(self.env, capacity=self.p.mfg_rigs)
        self.qc = simpy.Resource(self.env, capacity=self.p.qc_stations)
        # Clinic modeled as discrete daily slots: we'll keep a counter per day
        self.clinic_slots = {}
        self.events: List[Dict] = []  # event log rows

    # --- Event logging
    def log(self, patient_id: str, step: str, start: float, end: float):
        self.events.append({
            'patient_id': patient_id,
            'step': step,
            'start_day': start,
            'end_day': end,
            'duration_days': end - start
        })

    # --- Step execution wrapper
    def _do_step(self, res: simpy.Resource, duration_fn, patient_id: str, step_name: str):
        start = self.env.now
        with res.request() as req:
            yield req
            dur = duration_fn()
            yield self.env.timeout(dur)
        end = self.env.now
        self.log(patient_id, step_name, start, end)

    # --- Clinic slot scheduling: next available day with a free slot
    def _book_infusion_slot(self) -> float:
        day = math.floor(self.env.now)  # current day boundary or later
        while True:
            slots = self.clinic_slots.get(day, 0)
            if slots < self.p.clinic_slots_per_day:
                self.clinic_slots[day] = slots + 1
                # Infusion takes some (short) duration but is scheduled at this day
                return float(day)
            day += 1

    def patient_flow(self, idx: int):
        pid = f"P{idx:03d}"
        steps = self.p.steps

        # 1) Apheresis (collection)
        yield self.env.process(self._do_step(
            self.apheresis,
            lambda: truncated_gauss(steps['Apheresis'].mean, steps['Apheresis'].sd),
            pid, 'Apheresis'
        ))

        # 2) Ship-in to manufacturing site
        start = self.env.now
        dur_ship_in = truncated_gauss(steps['Ship-In'].mean, steps['Ship-In'].sd)
        yield self.env.timeout(dur_ship_in)
        self.log(pid, 'Ship-In', start, self.env.now)

        # 3) Manufacturing (autologous)
        yield self.env.process(self._do_step(
            self.mfg,
            lambda: truncated_gauss(steps['Manufacturing'].mean, steps['Manufacturing'].sd),
            pid, 'Manufacturing'
        ))

        # 4) Fill/Finish (small buffer step)
        start = self.env.now
        dur_ff = truncated_gauss(steps['Fill/Finish'].mean, steps['Fill/Finish'].sd)
        yield self.env.timeout(dur_ff)
        self.log(pid, 'Fill/Finish', start, self.env.now)

        # 5) QC (duration governed by qc_days)
        yield self.env.process(self._do_step(
            self.qc,
            lambda: truncated_gauss(self.p.qc_days, steps['QC'].sd),
            pid, 'QC'
        ))

        # 6) QP Release (administrative)
        start = self.env.now
        dur_rel = truncated_gauss(steps['Release'].mean, steps['Release'].sd)
        yield self.env.timeout(dur_rel)
        self.log(pid, 'Release', start, self.env.now)

        # 7) Ship-out to clinic
        start = self.env.now
        dur_ship_out = truncated_gauss(steps['Ship-Out'].mean, steps['Ship-Out'].sd)
        yield self.env.timeout(dur_ship_out)
        self.log(pid, 'Ship-Out', start, self.env.now)

        # 8) Infusion — assign next available clinic slot (day‑bucketed)
        #    We model a scheduling delay if same-day slots are exhausted.
        booked_day = self._book_infusion_slot()
        if booked_day > self.env.now:
            # wait until booked day
            yield self.env.timeout(booked_day - self.env.now)
        # perform infusion step
        start = self.env.now
        dur_inf = truncated_gauss(steps['Infusion'].mean, steps['Infusion'].sd)
        yield self.env.timeout(dur_inf)
        self.log(pid, 'Infusion', start, self.env.now)

    def run(self) -> pd.DataFrame:
        random.seed(self.p.seed)
        np.random.seed(self.p.seed)

        # Patient arrivals (apheresis start times)
        def arrival_process(env):
            for i in range(1, self.p.n_patients + 1):
                env.process(self.patient_flow(i))
                # Poisson-like arrivals via exponential interarrival
                gap = max(np.random.exponential(self.p.interarrival_days), 0.01)
                yield env.timeout(gap)

        self.env.process(arrival_process(self.env))
        # Run long enough; rough upper bound
        self.env.run(until=365.0)
        return pd.DataFrame(self.events)

# -------------------------
# KPI calculation & visuals
# -------------------------

def compute_kpis(df: pd.DataFrame, target_lead_time_days: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df.empty:
        return df, {
            'throughput': 0,
            'lead_time_median': float('nan'),
            'lead_time_p90': float('nan'),
            'on_time_pct': float('nan')
        }

    # Per-patient lead time: from start of Apheresis to end of Infusion
    lead = (df.pivot_table(index='patient_id',
                           values=['start_day', 'end_day'],
                           columns='step',
                           aggfunc={'start_day': 'min', 'end_day': 'max'}))
    # Flatten columns
    lead.columns = [f"{a}_{b}" for a, b in lead.columns]
    lead = lead.reset_index()

    # Derive overall start and end
    lead['t0'] = lead.filter(like='start_day').min(axis=1)
    lead['t_end'] = lead.filter(like='end_day_Infusion').fillna(method='ffill', axis=1)
    if 'end_day_Infusion' in df.columns:
        pass

    # safer: get infusion end by merge
    inf_end = df[df['step'] == 'Infusion'][['patient_id', 'end_day']].rename(columns={'end_day': 'inf_end'})
    aph_start = df[df['step'] == 'Apheresis'][['patient_id', 'start_day']].rename(columns={'start_day': 'aph_start'})
    lt = pd.merge(aph_start, inf_end, on='patient_id', how='inner')
    lt['lead_time'] = lt['inf_end'] - lt['aph_start']

    throughput = lt.shape[0]
    lead_time_median = float(lt['lead_time'].median()) if throughput > 0 else float('nan')
    lead_time_p90 = float(lt['lead_time'].quantile(0.90)) if throughput > 0 else float('nan')
    on_time_pct = float((lt['lead_time'] <= target_lead_time_days).mean() * 100.0) if throughput > 0 else float('nan')

    kpis = {
        'throughput': throughput,
        'lead_time_median': lead_time_median,
        'lead_time_p90': lead_time_p90,
        'on_time_pct': on_time_pct
    }
    return lt, kpis


def make_gantt(df_events: pd.DataFrame, ref_date: pd.Timestamp) -> px.timeline:
    if df_events.empty:
        return None
    # Convert day offsets to real datetimes for nicer Gantt
    df_plot = df_events.copy()
    df_plot['Start'] = ref_date + pd.to_timedelta(df_plot['start_day'], unit='D')
    df_plot['Finish'] = ref_date + pd.to_timedelta(df_plot['end_day'], unit='D')
    fig = px.timeline(
        df_plot,
        x_start='Start', x_end='Finish',
        y='patient_id', color='step',
        hover_data={'start_day': True, 'end_day': True, 'duration_days': ':.2f'}
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="CGT Supply Chain Simulator (Autologous v1)", layout="wide")
st.title("Cell & Gene Therapy Supply Chain Simulator — Autologous v1")

with st.sidebar:
    st.header("Scenario Parameters")
    n_patients = st.slider("# Patients", 5, 150, 40, step=5)
    interarrival = st.slider("Mean Interarrival (days)", 0.1, 5.0, 0.5, step=0.1)

    st.subheader("Capacities")
    apheresis_cap = st.slider("Apheresis Chairs (concurrent)", 1, 10, 3)
    mfg_rigs = st.slider("Manufacturing Rigs (concurrent)", 1, 10, 3)
    qc_stations = st.slider("QC Stations (concurrent)", 1, 10, 2)
    clinic_slots = st.slider("Clinic Infusion Slots / day", 1, 20, 6)

    st.subheader("Durations (means ± sd) — days")
    aph_mean = st.number_input("Apheresis mean", 0.1, 5.0, 0.5, step=0.1)
    aph_sd = st.number_input("Apheresis sd", 0.0, 2.0, 0.1, step=0.1)

    shipin_mean = st.number_input("Ship-In mean", 0.1, 5.0, 0.75, step=0.05)
    shipin_sd = st.number_input("Ship-In sd", 0.0, 2.0, 0.15, step=0.05)

    mfg_mean = st.number_input("Manufacturing mean", 1.0, 30.0, 7.0, step=0.5)
    mfg_sd = st.number_input("Manufacturing sd", 0.0, 10.0, 1.0, step=0.5)

    ff_mean = st.number_input("Fill/Finish mean", 0.1, 5.0, 0.5, step=0.1)
    ff_sd = st.number_input("Fill/Finish sd", 0.0, 2.0, 0.1, step=0.1)

    qc_mode = st.selectbox("QC Method", ["USP <71> Sterility (14 d)", "Rapid Method (3 d)", "Custom"], index=0)
    custom_qc = st.number_input("Custom QC days", 0.5, 20.0, 5.0, step=0.5, help="Only used if QC Method is 'Custom'")

    rel_mean = st.number_input("QP Release mean", 0.05, 5.0, 0.25, step=0.05)
    rel_sd = st.number_input("QP Release sd", 0.0, 2.0, 0.05, step=0.05)

    shipout_mean = st.number_input("Ship-Out mean", 0.1, 5.0, 0.75, step=0.05)
    shipout_sd = st.number_input("Ship-Out sd", 0.0, 2.0, 0.15, step=0.05)

    inf_mean = st.number_input("Infusion mean", 0.05, 2.0, 0.2, step=0.05)
    inf_sd = st.number_input("Infusion sd", 0.0, 1.0, 0.05, step=0.05)

    st.subheader("Targets & Reproducibility")
    target_lt = st.slider("On-Time Lead Time Target (days)", 7, 90, 28)
    seed = st.number_input("Random Seed", min_value=0, max_value=10_000, value=42)

# QC duration selection
if qc_mode.startswith("USP"):
    qc_days = 14.0
elif qc_mode.startswith("Rapid"):
    qc_days = 3.0
else:
    qc_days = float(custom_qc)

steps = {
    'Apheresis': StepParams(aph_mean, aph_sd),
    'Ship-In': StepParams(shipin_mean, shipin_sd),
    'Manufacturing': StepParams(mfg_mean, mfg_sd),
    'Fill/Finish': StepParams(ff_mean, ff_sd),
    'QC': StepParams(qc_days, 0.5),  # sd used for minor stochasticity around setpoint
    'Release': StepParams(rel_mean, rel_sd),
    'Ship-Out': StepParams(shipout_mean, shipout_sd),
    'Infusion': StepParams(inf_mean, inf_sd)
}

params = ScenarioParams(
    n_patients=int(n_patients),
    interarrival_days=float(interarrival),
    apheresis_capacity=int(apheresis_cap),
    mfg_rigs=int(mfg_rigs),
    qc_stations=int(qc_stations),
    clinic_slots_per_day=int(clinic_slots),
    steps=steps,
    qc_days=float(qc_days),
    target_lead_time_days=float(target_lt),
    seed=int(seed)
)

# Run button
col_run, col_note = st.columns([1, 3])
with col_run:
    run_sim = st.button("Run Simulation", type="primary")
with col_note:
    st.caption("Tip: Toggle QC method and adjust capacities to see vein‑to‑vein impact. Time unit = days.")

if run_sim:
    sim = CGTSim(params)
    df_events = sim.run()

    # KPIs
    lt_df, kpis = compute_kpis(df_events, params.target_lead_time_days)

    # KPI cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Throughput (completed infusions)", f"{kpis['throughput']}")
    m2.metric("Lead Time Median (d)", f"{kpis['lead_time_median']:.1f}" if not math.isnan(kpis['lead_time_median']) else "—")
    m3.metric("Lead Time P90 (d)", f"{kpis['lead_time_p90']:.1f}" if not math.isnan(kpis['lead_time_p90']) else "—")
    m4.metric("On‑Time Infusions (%)", f"{kpis['on_time_pct']:.0f}%" if not math.isnan(kpis['on_time_pct']) else "—")

    st.divider()

    # Timeline (Gantt)
    ref_date = pd.Timestamp('2025-01-01')
    fig = make_gantt(df_events, ref_date)
    if fig is not None:
        st.subheader("Patient Timelines (Gantt)")
        st.plotly_chart(fig, use_container_width=True)

    # Raw tables (expanders)
    with st.expander("Event Log (per step)"):
        st.dataframe(df_events.sort_values(['patient_id', 'start_day']).reset_index(drop=True), use_container_width=True)
    with st.expander("Per‑Patient Lead Times"):
        st.dataframe(lt_df.sort_values('patient_id').reset_index(drop=True), use_container_width=True)

    # CSV export
    st.subheader("Export Results")
    # Combine metadata for reproducibility
    meta = {
        'qc_method': qc_mode,
        'qc_days': params.qc_days,
        'target_lead_time_days': params.target_lead_time_days,
        'seed': params.seed,
        'capacities': {
            'apheresis': params.apheresis_capacity,
            'mfg_rigs': params.mfg_rigs,
            'qc_stations': params.qc_stations,
            'clinic_slots_per_day': params.clinic_slots_per_day
        }
    }
    # Prepare CSVs in-memory
    events_csv = df_events.to_csv(index=False).encode('utf-8')
    lt_csv = lt_df.to_csv(index=False).encode('utf-8')
    kpi_df = pd.DataFrame([{**kpis, **{'qc_days': params.qc_days, 'qc_method': qc_mode, 'target_lead_time_days': params.target_lead_time_days}}])
    kpi_csv = kpi_df.to_csv(index=False).encode('utf-8')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download Event Log CSV", events_csv, file_name="events.csv", mime="text/csv")
    with c2:
        st.download_button("Download Lead Times CSV", lt_csv, file_name="lead_times.csv", mime="text/csv")
    with c3:
        st.download_button("Download KPI Summary CSV", kpi_csv, file_name="kpis.csv", mime="text/csv")

    st.caption("Exported files are suitable for lesson handouts and further analysis (e.g., sensitivity charts).")

else:
    st.info("Set parameters in the sidebar, then click **Run Simulation**.")

st.divider()
st.markdown(
    """
    **About this prototype**
    - Autologous flow: Apheresis → Ship‑In → Manufacturing → Fill/Finish → QC → Release → Ship‑Out → Infusion.
    - Clinic slots are day‑bucketed to mimic real scheduling constraints.
    - QC toggle instantaneously changes the critical‑path duration (14 d vs 3 d vs custom).
    - Extendable modules (planned): Allogeneic batch planning, Vector capacity, Rapid‑QC vs USP what‑ifs.
    """
)
