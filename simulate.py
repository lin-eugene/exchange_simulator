"""Streamlit workbench for exchange-profile simulations.

Run with: streamlit run simulate.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from exchange_simulator.simulators.sim_all import SimulateExchange


st.set_page_config(page_title="Exchange simulator", page_icon="~", layout="wide")


def _profile_figure(x, y, xlabel, ylabel, title):
	figure, axis = plt.subplots(figsize=(8, 4.5))
	axis.plot(x, y, linewidth=2, color="#0f766e")
	axis.set_xlabel(xlabel)
	axis.set_ylabel(ylabel)
	axis.set_title(title)
	axis.grid(alpha=0.25)
	figure.tight_layout()
	return figure


st.title("Two-state exchange simulator")
st.caption("Explore CPMG, CEST, R2, R1rho, and cfast profiles from one parameter set.")

with st.sidebar:
	st.header("Exchange parameters")
	delta_omega = st.number_input("Delta omega (Hz)", value=100.0, min_value=0.0, step=10.0)
	delta_r2 = st.number_input("Delta R2 (s-1)", value=0.0, step=1.0)
	r2a = st.number_input("R2A (s-1)", value=10.0, min_value=0.0, step=1.0)
	r1a = st.number_input("R1A (s-1)", value=1.0, min_value=0.0, step=0.1)
	pb = st.number_input("Population B", value=0.10, min_value=0.0001, max_value=0.9999, step=0.01)
	kex = st.number_input("kex (s-1)", value=1000.0, min_value=0.0, step=50.0)
	omega_a = st.number_input("omegaA (Hz)", value=0.0, step=10.0)
	b0 = st.number_input("B0 (MHz)", value=95.0, min_value=0.0, step=1.0)

	st.divider()
	st.header("Acquisition settings")
	t_relax_cpmg = st.number_input("CPMG T_relax (s)", value=0.08, min_value=0.001, step=0.01)
	t_relax_cest = st.number_input("CEST T_relax (s)", value=0.70, min_value=0.001, step=0.05)
	t_relax_r1rho = st.number_input("R1rho T_relax (s)", value=0.16, min_value=0.001, step=0.01)
	omega1_cpmg = st.number_input("CPMG RF field omega1 (Hz)", value=6410.0, min_value=1.0, step=100.0)
	omega1_cest_values = st.multiselect(
		"CEST B1 fields omega1 (Hz)",
		options=[250.0, 500.0, 750.0],
		default=[250.0, 500.0, 750.0],
	)
	omega1_r1rho = st.number_input("R1rho RF field omega1 (Hz)", value=2000.0, min_value=1.0, step=100.0)
	points = st.slider("Points per profile", min_value=10, max_value=100, value=40, step=5)
	run = st.button("Simulate profiles", type="primary", use_container_width=True)


params = {
	"R2A": r2a,
	"R2B": r2a + delta_r2,
	"R1A": r1a,
	"R1B": r1a,
	"pb": pb,
	"kex": kex,
	"omegaA": omega_a,
	"omegaB": omega_a + delta_omega,
	"B0": b0,
}

if run or "profiles" not in st.session_state or "cest_y_by_b1" not in st.session_state.profiles:
	with st.spinner("Calculating profiles..."):
		simulator = SimulateExchange(params)
		cpmg_x = np.linspace(10.0, 2000.0, points)
		cest_x = np.linspace(-15000, 15000, points)
		r1rho_x = np.linspace(-4000, 4000, points)

		ncycs = np.maximum(1, np.rint(cpmg_x * t_relax_cpmg)).astype(int)
		cpmg_x, cpmg_y = simulator.simulate_CPMG_numerical(
			ncycs=ncycs,
			omega1=omega1_cpmg,
			T_relax=t_relax_cpmg,
		)
		cest_y_by_b1 = {
			omega1: np.array(
				simulator.simulate_CEST(cest_x, omega1=omega1, time_CEST=t_relax_cest)
			)
			for omega1 in omega1_cest_values
		}
		r1rho_profile = simulator.simulate_R1rho(r1rho_x, T_relax=t_relax_r1rho, omega1=omega1_r1rho)
		st.session_state.profiles = {
			"cpmg_x": cpmg_x,
			"cpmg_y": np.array(cpmg_y),
			"cest_x": cest_x,
			"cest_y_by_b1": cest_y_by_b1,
			"r1rho_x": r1rho_x,
			"r1rho_y": np.array(list(r1rho_profile.R1rhos.values())),
			"cfast_y": np.array(list(r1rho_profile.cfasts.values())),
			"r2_obs": simulator.deltaR2,
			"params": params,
		}

profiles = st.session_state.profiles
st.metric("Calculated exchange contribution, delta R2", f"{profiles['r2_obs']:.3f} s-1")

r2_values = [profiles["params"]["R2A"], profiles["params"]["R2B"], profiles["params"]["R2A"] + profiles["r2_obs"]]
figure, axes = plt.subplots(3, 2, figsize=(14, 11))
axes = axes.ravel()
r2_time = np.linspace(0.0, 0.5, 100)

axes[0].plot(profiles["cpmg_x"], profiles["cpmg_y"], linewidth=2, color="#0f766e")
axes[0].set_xlabel("CPMG frequency (Hz)")
axes[0].set_ylabel("R2eff (s-1)")
axes[0].set_title("CPMG relaxation dispersion")

colors = ["#2563eb", "#7c3aed", "#db2777"]
for color, (omega1, cest_y) in zip(colors, profiles["cest_y_by_b1"].items()):
	axes[1].plot(profiles["cest_x"], cest_y, linewidth=2, color=color, label=f"B1 = {omega1:.0f} Hz")
axes[1].set_xlabel("RF offset (Hz)")
axes[1].set_ylabel("I / I0")
axes[1].set_title("CEST profile")
axes[1].legend()

axes[2].plot(r2_time, np.exp(-r2_values[0] * r2_time), linewidth=2, label="A", color="#0f766e")
axes[2].plot(r2_time, np.exp(-r2_values[1] * r2_time), linewidth=2, label="B", color="#2563eb")
axes[2].plot(r2_time, np.exp(-r2_values[2] * r2_time), linewidth=2, label="Observed", color="#f97316")
axes[2].set_xlabel("Relaxation time (s)")
axes[2].set_ylabel("Normalized intensity")
axes[2].set_title("Transverse relaxation profile")
axes[2].legend()

axes[3].bar(["R2A", "R2B", "R2obs"], r2_values, color=["#0f766e", "#2563eb", "#f97316"])
axes[3].set_xlabel("Component")
axes[3].set_ylabel("Rate (s-1)")
axes[3].set_title("Transverse relaxation rates")

axes[4].plot(profiles["r1rho_x"], profiles["r1rho_y"], linewidth=2, color="#0f766e")
axes[4].set_xlabel("RF offset (Hz)")
axes[4].set_ylabel("R1rho (s-1)")
axes[4].set_title("R1rho profile")

axes[5].plot(profiles["r1rho_x"], profiles["cfast_y"], linewidth=2, color="#f97316")
axes[5].set_xlabel("RF offset (Hz)")
axes[5].set_ylabel("cfast")
axes[5].set_title("cfast profile")

for axis in axes:
	axis.grid(alpha=0.25)

figure.tight_layout()
st.pyplot(figure, clear_figure=True)
st.write(f"Input delta R2: {delta_r2:.3f} s-1")
