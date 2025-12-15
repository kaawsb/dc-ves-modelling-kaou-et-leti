# ==============================================================
st.error("✅ VERSION LATEX FIX — 15/12")
# 1D DC Forward Modelling (SimPEG)
# Schlumberger & Wenner — Application Streamlit pédagogique
#
# Objectif :
# - Simuler un sondage électrique vertical (VES)
# - Comparer Schlumberger et Wenner
# - Visualiser les courbes rho_a(AB/2)
# - Relier les résultats à la théorie
# ==============================================================


# ==============================================================
# 0) IMPORTS
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter


# ==============================================================
# 1) CONFIGURATION STREAMLIT
# ==============================================================

st.set_page_config(
    page_title="1D DC VES — Schlumberger & Wenner",
    page_icon="🌍",
    layout="wide",
)

st.title("1D DC Resistivity — Schlumberger vs Wenner (SimPEG)")
st.markdown(
    """
    Simulation de **sondage électrique vertical (VES)** en milieu **1D**.  
    L’application calcule la **résistivité apparente** ρₐ en fonction de **AB/2**
    pour les dispositifs **Schlumberger** et **Wenner**.
    """
)

st.divider()


# ==============================================================
# 2) FONCTIONS UTILITAIRES
# ==============================================================

def build_log_ticks(ax):
    """Axes logarithmiques propres (graduations principales et secondaires)."""
    major = LogLocator(base=10.0, subs=(1.0,))
    minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_major_locator(major)
    ax.yaxis.set_minor_locator(minor)

    ax.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=True))
    ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=True))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def compute_rho_limits(*arrays):
    """Bornes propres pour l’axe Y (log)."""
    vals = np.hstack(arrays)
    ymin = 10 ** np.floor(np.log10(vals.min()))
    ymax = 10 ** np.ceil(np.log10(vals.max()))
    return ymin, ymax


def make_schlumberger_survey(AB2, MN2):
    """Construction du dispositif Schlumberger."""
    src_list = []
    eps = 1e-6

    for L, a in zip(AB2, MN2):
        A = np.r_[-L, 0.0, 0.0]
        B = np.r_[+L, 0.0, 0.0]
        M = np.r_[-(a - eps), 0.0, 0.0]
        N = np.r_[+(a - eps), 0.0, 0.0]

        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
        src = dc.sources.Dipole([rx], A, B)
        src_list.append(src)

    return dc.Survey(src_list)


def make_wenner_survey(AB2):
    """Construction du dispositif Wenner."""
    src_list = []

    for L in AB2:
        a = (2.0 / 3.0) * L
        A = np.r_[-1.5*a, 0.0, 0.0]
        M = np.r_[-0.5*a, 0.0, 0.0]
        N = np.r_[+0.5*a, 0.0, 0.0]
        B = np.r_[+1.5*a, 0.0, 0.0]

        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
        src = dc.sources.Dipole([rx], A, B)
        src_list.append(src)

    return dc.Survey(src_list)


def run_forward(survey, rho_layers, thicknesses):
    """Modélisation directe SimPEG."""
    rho_map = maps.IdentityMap(nP=len(rho_layers))
    sim = dc.simulation_1d.Simulation1DLayers(
        survey=survey,
        rhoMap=rho_map,
        thicknesses=thicknesses,
    )
    return sim.dpred(rho_layers)


# ==============================================================
# 3) SIDEBAR — PARAMÈTRES
# ==============================================================

with st.sidebar:

    st.header("Géométrie (AB/2)")

    ab2_min = st.number_input("AB/2 min (m)", 0.1, value=5.0)
    ab2_max = st.number_input("AB/2 max (m)", ab2_min+0.1, value=300.0)

    n_stations = st.slider("Nombre de stations", 8, 60, 25)

    st.caption(
        """
        Schlumberger : MN/2 = 0.1·AB/2  
        Wenner : AB = 3a, MN = a
        """
    )

    st.divider()
    st.header("Modèle de couches")

    n_layers = st.slider("Nombre de couches", 3, 5, 4)

    default_rho = [10, 30, 15, 50, 100][:n_layers]
    default_thk = [2, 8, 60, 120][:n_layers-1]

    rho_layers = []
    for i in range(n_layers):
        rho_layers.append(
            st.number_input(f"ρ couche {i+1} (Ω·m)", 0.1, value=float(default_rho[i]))
        )

    thicknesses = []
    for i in range(n_layers-1):
        thicknesses.append(
            st.number_input(f"Épaisseur couche {i+1} (m)", 0.1, value=float(default_thk[i]))
        )


rho_layers = np.array(rho_layers)
thicknesses = np.array(thicknesses)

st.divider()


# ==============================================================
# 4) CALCULS
# ==============================================================

AB2 = np.geomspace(ab2_min, ab2_max, n_stations)
MN2 = np.minimum(0.10 * AB2, 0.49 * AB2)

survey_s = make_schlumberger_survey(AB2, MN2)
survey_w = make_wenner_survey(AB2)

rho_app_s = run_forward(survey_s, rho_layers, thicknesses)
rho_app_w = run_forward(survey_w, rho_layers, thicknesses)


# ==============================================================
# 5) ONGLET COURBES
# ==============================================================

tab1, tab2, tab3 = st.tabs(["📈 Courbes", "🧱 Modèle", "📚 Théorie"])

with tab1:

    fig, ax = plt.subplots(figsize=(7,5))
    ax.loglog(AB2, rho_app_s, "o-", label="Schlumberger ρₐ")
    ax.loglog(AB2, rho_app_w, "s--", label="Wenner ρₐ")

    ymin, ymax = compute_rho_limits(rho_app_s, rho_app_w)
    ax.set_ylim(ymin, ymax)

    build_log_ticks(ax)
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.set_xlabel("AB/2 (m)")
    ax.set_ylabel("ρₐ (Ω·m)")
    ax.legend()

    st.pyplot(fig)


# ==============================================================
# 6) ONGLET THÉORIE — VERSION QUI MARCHE
# ==============================================================

with tab3:

    st.subheader("Rappels théoriques")

    st.markdown("### Loi d’Ohm (scalaire)")
    st.latex(r"U = R\,I")

    st.markdown("### Résistivité")
    st.latex(r"\rho = R\,\frac{A}{L}")
    st.markdown("Unité : Ω·m")

    st.markdown("### Conductivité")
    st.latex(r"\sigma = \frac{1}{\rho}")
    st.markdown("Unité : S/m")

    st.markdown("### Loi d’Ohm (vectorielle)")
    st.latex(r"\vec{J} = \sigma\,\vec{E}")

    st.markdown("### VES (Vertical Electrical Sounding)")
    st.markdown("- AB/2 ↑ → profondeur d’investigation ↑")
    st.markdown("- Mesure de ρₐ(AB/2)")

    st.markdown("### Dispositif Schlumberger")
    st.markdown("- A, B : électrodes de courant")
    st.markdown("- M, N : électrodes de potentiel proches du centre")

    st.markdown("### Dispositif Wenner")
    st.latex(r"AB = 3a \qquad MN = a")
