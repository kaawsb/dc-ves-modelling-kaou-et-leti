# ==============================================================
# 1D DC Forward Modelling (SimPEG) — Schlumberger + Wenner
# Version "optimisée & pédagogique" avec interface Streamlit
#
# Ce fichier réalise un sondage électrique vertical (1D VES)
# en utilisant la méthode du courant continu (DC resistivity).
# Il simule deux dispositifs classiques :
#   - Schlumberger
#   - Wenner
# Et trace les courbes de résistivité apparente rho_a(AB/2)
# pour un modèle de couches donné (rho + épaisseurs).
#
# Tout est expliqué en détail dans les commentaires (#)
# ==============================================================


# ==============================================================
# 0) IMPORTS DES LIBRAIRIES
# ==============================================================

import numpy as np                 # Calcul scientifique (tableaux + log + géométrique)
import pandas as pd               # Tableaux de données (pour afficher et exporter en CSV)
import matplotlib.pyplot as plt   # Graphiques (courbes, modèle 1D)
import streamlit as st            # Interface web Streamlit

# Import du module DC Resistivity de SimPEG (électrostatique)
from simpeg.electromagnetics.static import resistivity as dc

# maps.IdentityMap permet de dire : "mon modèle = ma résistivité directement"
from simpeg import maps

# Outils pour axes logarithmiques propres
from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter


# ==============================================================
# 1) CONFIGURATION STREAMLIT (DOIT ÊTRE TOUT EN HAUT)
# ==============================================================
st.set_page_config(
    page_title="1D DC VES — Schlumberger & Wenner",
    page_icon="🌍",
    layout="wide",
)


# ==============================================================
# 2) STYLE CSS — personnalisation visuelle de Streamlit
# ==============================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f8fb;
    }
    h1, h2, h3 {
        color: #1f4e79;
    }
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
        border-right: 1px solid #d5d8dd;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# 3) TITRE + INTRO
# ==============================================================
st.title("1D DC Resistivity — Schlumberger vs Wenner (SimPEG)")
st.markdown(
    """
    Simulation de sondage électrique vertical (**VES**, Vertical Electrical Sounding).  
    On calcule la **résistivité apparente** ρₐ en fonction de **AB/2** pour :
    - le dispositif **Schlumberger**
    - le dispositif **Wenner**
    """
)

st.divider()


# ==============================================================
# 4) FONCTIONS UTILITAIRES
# ==============================================================
def build_log_ticks(ax):
    """Configure les axes x et y en échelle logarithmique."""
    major_locator = LogLocator(base=10.0, subs=(1.0,))
    minor_locator = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)

    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
    ax.xaxis.set_minor_formatter(NullFormatter())


def compute_rho_limits(*arrays):
    """Bornes propres pour l’axe Y (log)."""
    vals = np.hstack(arrays)
    ymin = 10 ** np.floor(np.log10(vals.min()))
    ymax = 10 ** np.ceil(np.log10(vals.max()))
    return ymin, ymax


def build_layer_interfaces(thicknesses):
    """Interfaces : [0, cumsum(thk)]."""
    if len(thicknesses) == 0:
        return np.array([0.0])
    return np.r_[0.0, np.cumsum(thicknesses)]


def make_schlumberger_survey(AB2, MN2):
    """Construction du dispositif Schlumberger."""
    src_list = []
    eps = 1e-6  # évite que M = N exactement

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
    """Construction du dispositif Wenner : AB = 3a et a = (2/3)*AB/2."""
    src_list = []

    for L in AB2:
        a = (2.0 / 3.0) * L
        A = np.r_[-1.5 * a, 0.0, 0.0]
        M = np.r_[-0.5 * a, 0.0, 0.0]
        N = np.r_[+0.5 * a, 0.0, 0.0]
        B = np.r_[+1.5 * a, 0.0, 0.0]

        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
        src = dc.sources.Dipole([rx], A, B)
        src_list.append(src)

    return dc.Survey(src_list)


def run_forward_1d(survey, rho_layers, thicknesses):
    """Forward 1D SimPEG."""
    rho_map = maps.IdentityMap(nP=len(rho_layers))
    sim = dc.simulation_1d.Simulation1DLayers(
        survey=survey,
        rhoMap=rho_map,
        thicknesses=thicknesses,
    )
    try:
        data = sim.dpred(rho_layers)
        return data, None
    except Exception as e:
        return None, str(e)


def plot_layer_model(rho_layers, thicknesses, ax=None):
    """Trace le modèle 1D en couches."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig = ax.figure

    n_layers = len(rho_layers)
    interfaces = build_layer_interfaces(thicknesses)
    z_bottom = interfaces[-1] + max(interfaces[-1] * 0.3, 10.0)

    tops = np.r_[interfaces, interfaces[-1]]
    bottoms = np.r_[interfaces[1:], z_bottom]

    colors = ["#f4b183", "#bdd7ee", "#c5e0b4", "#ffe699", "#e2b9ff"]

    for i in range(n_layers):
        ax.fill_betweenx([tops[i], bottoms[i]], 0, rho_layers[i],
                         color=colors[i % len(colors)], alpha=0.6)
        ax.text(
            rho_layers[i] * 1.05,
            (tops[i] + bottoms[i]) / 2,
            f"{rho_layers[i]:.1f} Ω·m",
            va="center",
            fontsize=9,
        )

    ax.invert_yaxis()
    ax.set_xlabel("Resistivity (Ω·m)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_title("Layered model")

    return fig, ax


# ==============================================================
# 5) BARRE LATÉRALE : paramètres du modèle et géométrie
# ==============================================================
with st.sidebar:
    st.header("Géométrie (AB/2)")

    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input("AB/2 min (m)", min_value=0.1, value=5.0, step=0.1)
    with colA2:
        ab2_max = st.number_input("AB/2 max (m)", min_value=ab2_min + 0.1, value=300.0, step=1.0)

    n_stations = st.slider("Nombre de stations", min_value=8, max_value=60, value=25)

    st.caption(
        """
        **Schlumberger :** MN/2 = 0.1 × AB/2 (limité à 0.49 × AB/2)  
        **Wenner :** AB = 3a, MN = a
        """
    )

    st.divider()
    st.header("Modèle de couches")

    scenario = st.selectbox(
        "Scénario géologique",
        [
            "Personnalisé",
            "Couche conductrice superficielle",
            "Nappe salée profonde",
            "Couche résistante sur substratum conducteur",
        ],
    )

    n_layers = st.slider("Nombre de couches", 3, 5, 4)

    if scenario == "Couche conductrice superficielle":
        default_rho = [5, 30, 80, 200, 300][:n_layers]
        default_thk = [3, 10, 40, 100][:n_layers - 1]
    elif scenario == "Nappe salée profonde":
        default_rho = [80, 60, 10, 15, 50][:n_layers]
        default_thk = [5, 15, 60, 120][:n_layers - 1]
    elif scenario == "Couche résistante sur substratum conducteur":
        default_rho = [20, 100, 15, 10, 8][:n_layers]
        default_thk = [10, 30, 80, 150][:n_layers - 1]
    else:
        default_rho = [10, 30, 15, 50, 100][:n_layers]
        default_thk = [2, 8, 60, 120][:n_layers - 1]

    st.subheader("ρ des couches (Ω·m)")
    layer_rhos = []
    for i in range(n_layers):
        layer_rhos.append(
            st.number_input(
                f"ρ couche {i+1} (Ω·m)",
                min_value=0.1,
                value=float(default_rho[i]),
                step=0.1,
            )
        )

    thicknesses = []
    st.caption("Épaisseurs des premières couches (la dernière est un demi-espace).")
    for i in range(n_layers - 1):
        thicknesses.append(
            st.number_input(
                f"Épaisseur couche {i+1} (m)",
                min_value=0.1,
                value=float(default_thk[i]),
                step=0.1,
            )
        )

rho_layers = np.r_[layer_rhos]
thicknesses = np.r_[thicknesses] if len(thicknesses) else np.array([])

st.divider()


# ==============================================================
# 6) Construction des surveys + Forward
# ==============================================================
AB2 = np.geomspace(ab2_min, ab2_max, n_stations)
MN2 = np.minimum(0.10 * AB2, 0.49 * AB2)

survey_schl = make_schlumberger_survey(AB2, MN2)
survey_wenn = make_wenner_survey(AB2)

rho_app_s, err_s = run_forward_1d(survey_schl, rho_layers, thicknesses)
rho_app_w, err_w = run_forward_1d(survey_wenn, rho_layers, thicknesses)

if err_s or err_w:
    st.error(f"Erreur Schlumberger: {err_s}\nErreur Wenner: {err_w}")
    st.stop()


# ==============================================================
# 7) ONGLET PRINCIPAL
# ==============================================================
tab_curves, tab_model, tab_theory = st.tabs(
    ["📈 Courbes", "🧱 Modèle & données", "📚 Théorie rapide"]
)

with tab_curves:
    st.subheader("Courbes de sondage (log–log)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(AB2, rho_app_s, "o-", label="Schlumberger ρₐ")
    ax.loglog(AB2, rho_app_w, "s--", label="Wenner ρₐ")

    ymin, ymax = compute_rho_limits(rho_app_s, rho_app_w)
    ax.set_ylim(ymin, ymax)

    build_log_ticks(ax)
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.set_xlabel("AB/2 (m)")
    ax.set_ylabel("ρₐ (Ω·m)")
    ax.set_title("Schlumberger vs Wenner — 1D VES (forward)")
    ax.legend()

    st.pyplot(fig, clear_figure=True)

    df_out = pd.DataFrame({
        "AB/2 (m)": AB2,
        "MN/2 Schl. (m)": MN2,
        "ρa Schl. (Ω·m)": rho_app_s,
        "ρa Wenn. (Ω·m)": rho_app_w,
    })
    st.download_button(
        "⬇️ Télécharger CSV",
        data=df_out.to_csv(index=False).encode(),
        file_name="ves_schl_wenn.csv",
        mime="text/csv",
    )

with tab_model:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Modèle 1D")
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        plot_layer_model(rho_layers, thicknesses, ax=ax2)
        st.pyplot(fig2, clear_figure=True)

    with col2:
        st.subheader("Tableau des couches")
        df_model = pd.DataFrame({
            "Couche": np.arange(1, len(rho_layers) + 1),
            "Resistivity (Ω·m)": rho_layers,
            "Thickness (m)": [*thicknesses, np.nan],
            "Note": [""] * (len(rho_layers) - 1) + ["Half-space"],
        })
        st.dataframe(df_model, use_container_width=True)

with tab_theory:
    st.subheader("Rappels théoriques")

    st.markdown("### Loi d’Ohm (scalaire)")
    st.latex(r"U = R\,I")

    st.markdown("### Résistivité")
    st.latex(r"\rho = R\,\frac{A}{L}")
    st.markdown("- Unité : **Ω·m**")

    st.markdown("### Conductivité")
    st.latex(r"\sigma = \frac{1}{\rho}")
    st.markdown("- Unité : **S/m**")

    st.markdown("### Loi d’Ohm vectorielle")
    st.latex(r"\vec{J} = \sigma\,\vec{E}")

    st.markdown("### VES (Vertical Electrical Sounding)")
    st.markdown("- AB/2 ↑ → profondeur d’investigation ↑")
    st.markdown("- On mesure ρₐ(AB/2) pour voir les changements de couches")

    st.markdown("### Dispositif Schlumberger")
    st.markdown("- A et B : électrodes de courant")
    st.markdown("- M et N : électrodes de potentiel proches du centre (MN/2 ≪ AB/2)")

    st.markdown("### Dispositif Wenner")
    st.markdown("- 4 électrodes régulièrement espacées")
    st.latex(r"AB = 3a \qquad MN = a")

st.caption(
    "Astuce : change le scénario dans la sidebar pour voir l’effet d’une couche conductrice ou résistante."
)
