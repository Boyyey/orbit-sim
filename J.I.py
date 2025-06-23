import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# Simulation constants
G = 6.6743e-11
AU = 1.5e11  # meters

st.set_page_config(page_title="3D Planetary Simulation", layout="wide", initial_sidebar_state="expanded")
st.title("🌌 3D Planetary & Black Hole Simulation")
st.markdown("""
<style>
body { background: #181a1b !important; }
[data-testid="stSidebar"] { background: #23272e; }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
def get_color(color_name):
    color_map = {
        "Cyan": "#00FFFF",
        "Yellow": "#FFFF00",
        "Red": "#FF0000",
        "Green": "#00FF00",
        "White": "#FFFFFF",
        "Magenta": "#FF00FF",
        "Orange": "#FFA500",
        "Black": "#222222"
    }
    return color_map.get(color_name, "#00FFFF")

st.sidebar.header("Add or Edit Body")
name = st.sidebar.text_input("Name", "Earth")
type_body = st.sidebar.selectbox("Type", ["Planet", "Black Hole"])
color_name = st.sidebar.selectbox("Color", ["Cyan", "Yellow", "Red", "Green", "White", "Magenta", "Orange", "Black"])
mass = st.sidebar.number_input("Mass (kg)", value=5.972e24, format="%.3e")
radius = st.sidebar.number_input("Radius (m)", value=6.4e7 if type_body=="Planet" else 1e10, format="%.3e")
col = get_color(color_name)

st.sidebar.markdown("**Position (meters):**")
x = st.sidebar.number_input("x", value=AU)
y = st.sidebar.number_input("y", value=0.0)
z = st.sidebar.number_input("z", value=0.0)

st.sidebar.markdown("**Velocity (m/s):**")
vx = st.sidebar.number_input("vx", value=0.0)
vy = st.sidebar.number_input("vy", value=29780.0)
vz = st.sidebar.number_input("vz", value=0.0)

if 'bodies' not in st.session_state:
    st.session_state.bodies = [
        {"name": "Sun", "type": "Planet", "color": "#FFFF00", "mass": 1.989e30, "radius": 1e11, "pos": [0,0,0], "vel": [0,0,0]},
        {"name": "Earth", "type": "Planet", "color": "#00FFFF", "mass": 5.972e24, "radius": 5e10, "pos": [AU,0,0], "vel": [0,29780,0]},
        {"name": "Earth2", "type": "Planet", "color": "#00FFFF", "mass": 5.972e24, "radius": 5e10, "pos": [-AU,0,0], "vel": [0,-29780,0]},
    ]

if st.sidebar.button("Add Body"):
    st.session_state.bodies.append({
        "name": name,
        "type": type_body,
        "color": col,
        "mass": mass,
        "radius": radius,
        "pos": [x, y, z],
        "vel": [vx, vy, vz]
    })
    st.sidebar.success(f"Added {name}")

if st.sidebar.button("Reset Bodies"):
    st.session_state.bodies = [
        {"name": "Sun", "type": "Planet", "color": "#FFFF00", "mass": 1.989e30, "radius": 1e11, "pos": [0,0,0], "vel": [0,0,0]},
        {"name": "Earth", "type": "Planet", "color": "#00FFFF", "mass": 5.972e24, "radius": 5e10, "pos": [AU,0,0], "vel": [0,29780,0]},
        {"name": "Earth2", "type": "Planet", "color": "#00FFFF", "mass": 5.972e24, "radius": 5e10, "pos": [-AU,0,0], "vel": [0,-29780,0]},
    ]
    st.sidebar.info("Reset to default bodies.")

# Simulation controls
st.sidebar.header("Simulation Controls")
time_step = st.sidebar.slider("Time Step (s)", 100, 100000, 3600, step=100)
run_sim = st.sidebar.button("Run Simulation Step")

# Add run/pause toggle
if 'running' not in st.session_state:
    st.session_state.running = False

run_col, pause_col = st.sidebar.columns(2)
if run_col.button("▶️ Run", use_container_width=True):
    st.session_state.running = True
if pause_col.button("⏸️ Pause", use_container_width=True):
    st.session_state.running = False

# Physics engine
def compute_forces(bodies):
    n = len(bodies)
    forces = [np.zeros(3) for _ in range(n)]
    for i, b1 in enumerate(bodies):
        for j, b2 in enumerate(bodies):
            if i == j:
                continue
            r_vec = np.array(b2["pos"]) - np.array(b1["pos"])
            dist = np.linalg.norm(r_vec)
            if dist < 1e-2:
                continue
            f = G * b1["mass"] * b2["mass"] / (dist ** 2)
            forces[i] += f * r_vec / dist
    return forces

def update_bodies(bodies, forces, dt):
    for b, f in zip(bodies, forces):
        acc = f / b["mass"]
        b["vel"] = list(np.array(b["vel"]) + acc * dt)
        b["pos"] = list(np.array(b["pos"]) + np.array(b["vel"]) * dt)

def velocity_verlet_step(bodies, dt):
    n = len(bodies)
    positions = [np.array(b["pos"]) for b in bodies]
    velocities = [np.array(b["vel"]) for b in bodies]
    masses = [b["mass"] for b in bodies]
    # Compute initial accelerations
    accels = []
    for i, b1 in enumerate(bodies):
        a = np.zeros(3)
        for j, b2 in enumerate(bodies):
            if i == j:
                continue
            r_vec = positions[j] - positions[i]
            dist = np.linalg.norm(r_vec)
            if dist < 1e-2:
                continue
            a += G * masses[j] * r_vec / dist**3
        accels.append(a)
    # Update positions
    new_positions = [pos + v*dt + 0.5*a*dt**2 for pos, v, a in zip(positions, velocities, accels)]
    # Compute new accelerations
    new_accels = []
    for i, b1 in enumerate(bodies):
        a = np.zeros(3)
        for j, b2 in enumerate(bodies):
            if i == j:
                continue
            r_vec = new_positions[j] - new_positions[i]
            dist = np.linalg.norm(r_vec)
            if dist < 1e-2:
                continue
            a += G * masses[j] * r_vec / dist**3
        new_accels.append(a)
    # Update velocities
    new_velocities = [v + 0.5*(a1+a2)*dt for v, a1, a2 in zip(velocities, accels, new_accels)]
    # Write back
    for i, b in enumerate(bodies):
        b["pos"] = list(new_positions[i])
        b["vel"] = list(new_velocities[i])

# Run simulation step
if run_sim:
    forces = compute_forces(st.session_state.bodies)
    update_bodies(st.session_state.bodies, forces, time_step)
    st.success("Simulation step completed.")

# Real-time simulation loop
if st.session_state.running:
    velocity_verlet_step(st.session_state.bodies, time_step)
    time.sleep(0.05)  # ~20 FPS
    st.rerun()

# 3D Visualization
fig = go.Figure()
for b in st.session_state.bodies:
    fig.add_trace(go.Scatter3d(
        x=[b["pos"][0]], y=[b["pos"][1]], z=[b["pos"][2]],
        mode='markers',
        marker=dict(size=max(8, b["radius"] / 1e10), color=b["color"], opacity=0.9 if b["type"]=="Planet" else 0.6, symbol='circle'),
        name=b["name"] + (" (Black Hole)" if b["type"]=="Black Hole" else "")
    ))
fig.update_layout(
    scene=dict(
        xaxis=dict(title='x (m)', backgroundcolor="#222", color="#fff"),
        yaxis=dict(title='y (m)', backgroundcolor="#222", color="#fff"),
        zaxis=dict(title='z (m)', backgroundcolor="#222", color="#fff"),
        bgcolor="#181a1b"
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    paper_bgcolor="#181a1b",
    font_color="#fff",
    legend=dict(bgcolor="#23272e", font=dict(color="#fff"))
)
st.plotly_chart(fig, use_container_width=True)

# List bodies
with st.expander("Show All Bodies"):
    for b in st.session_state.bodies:
        st.write(b)
