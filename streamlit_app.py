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
# 1) STYLE CSS — personnalisation visuelle de Streamlit
# ==============================================================

# On injecte du CSS pour rendre l'application plus agréable à l'œil.
# Ce bloc modifie :
#   - la couleur de fond
#   - l'apparence de la sidebar
#   - la présentation des tableaux Streamlit
st.markdown(
    """
    <style>
    .stApp {
        /* Couleur d’arrière-plan générale */
        background-color: #f7f8fb;
    }

    h1, h2, h3 {
        /* Couleur des titres */
        color: #1f4e79;
    }

    /* Style de la barre latérale */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
        border-right: 1px solid #d5d8dd;
    }

    /* Style pour les DataFrames Streamlit */
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
# 2) FONCTIONS UTILITAIRES
# ==============================================================

# --------------------------------------------------------------
# Formatage des axes log-log : rend les graduations propres
# --------------------------------------------------------------
def build_log_ticks(ax):
    """Configure les axes x et y en échelle logarithmique."""
    major_locator = LogLocator(base=10.0, subs=(1.0,))         # Ticks principaux : 1, 10, 100
    minor_locator = LogLocator(base=10.0, subs=np.arange(2,10)*0.1) # Petits ticks : 2...9

    # Axe Y
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Axe X
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
    ax.xaxis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------
# Calcule des limites verticales pour l’axe Y (log)
# --------------------------------------------------------------
def compute_rho_limits(*arrays):
    """
    On prend toutes les valeurs des courbes (Schlumberger + Wenner),
    puis on arrondit au-dessus et en dessous à la puissance de 10 la plus proche.
    """
    vals = np.hstack(arrays)
    ymin = vals.min()
    ymax = vals.max()

    ymin = 10 ** np.floor(np.log10(ymin))   # Ex : 14 → 10
    ymax = 10 ** np.ceil(np.log10(ymax))    # Ex : 73 → 100

    return ymin, ymax


# --------------------------------------------------------------
# Renvoie les profondeurs d’interface des couches
# --------------------------------------------------------------
def build_layer_interfaces(thicknesses):
    """
    thicknesses = liste des épaisseurs des couches 1..N-1.
    Exemple : [5, 20, 50] → interfaces = [0, 5, 25, 75]
    """
    if len(thicknesses) == 0:
        return np.array([0.0])
    return np.r_[0.0, np.cumsum(thicknesses)]


# --------------------------------------------------------------
# Construction du dispositif Schlumberger pour SimPEG
# --------------------------------------------------------------
def make_schlumberger_survey(AB2, MN2):
    """
    Un dispositif Schlumberger possède :
      - électrodes de courant A et B écartées de 2·AB/2
      - électrodes de potentiel M et N proches du centre

    Ici,
      - AB2 est un tableau avec toutes les valeurs AB/2 utilisées
      - MN2 = 0.1 * AB2 (ou limité)

    SimPEG demande :
      - la position de A, B, M, N dans le plan (x,y,z)
    """
    src_list = []
    eps = 1e-6  # évite que M = N exactement

    for L, a in zip(AB2, MN2):
        # A et B → injecteurs de courant
        A = np.r_[-L, 0.0, 0.0]    # -L sur l’axe x
        B = np.r_[+L, 0.0, 0.0]    # +L

        # M et N → électrodes de potentiel (proche du centre)
        M = np.r_[-(a - eps), 0.0, 0.0]
        N = np.r_[+(a - eps), 0.0, 0.0]

        # Récepteur SimPEG → dipôle MN
        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")

        # Source SimPEG → dipôle AB
        src = dc.sources.Dipole([rx], A, B)

        src_list.append(src)

    return dc.Survey(src_list)


# --------------------------------------------------------------
# Construction du dispositif Wenner pour SimPEG
# --------------------------------------------------------------
def make_wenner_survey(AB2):
    """
    Wenner : espacement régulier a entre A-M-N-B :
      A --- M --- N --- B
      AB = 3a → AB/2 = 1.5a → a = (2/3)*AB2
    """
    src_list = []

    for L in AB2:
        a = (2.0 / 3.0) * L

        # positions des électrodes
        A = np.r_[-1.5*a, 0.0, 0.0]
        M = np.r_[-0.5*a, 0.0, 0.0]
        N = np.r_[+0.5*a, 0.0, 0.0]
        B = np.r_[+1.5*a, 0.0, 0.0]

        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
        src = dc.sources.Dipole([rx], A, B)
        src_list.append(src)

    return dc.Survey(src_list)


# --------------------------------------------------------------
# Exécution de la modélisation directe (forward) avec SimPEG
# --------------------------------------------------------------
def run_forward_1d(survey, rho_layers, thicknesses):
    """
    SimPEG Simulation1DLayers calcule :
        rho_a(AB/2)
    pour le dispositif donné.

    rho_layers = résistivité de chaque couche (Ω·m)
    thicknesses = épaisseurs des N-1 premières couches
    """
    rho_map = maps.IdentityMap(nP=len(rho_layers))  # modèle = rho directement

    sim = dc.simulation_1d.Simulation1DLayers(
        survey=survey,
        rhoMap=rho_map,
        thicknesses=thicknesses,
    )

    try:
        data = sim.dpred(rho_layers)  # prédiction de données
        return data, None
    except Exception as e:
        return None, str(e)


# --------------------------------------------------------------
# Tracé du modèle 1D en couches
# --------------------------------------------------------------
def plot_layer_model(rho_layers, thicknesses, ax=None):
    """
    Trace verticalement :
        - chaque couche avec sa résistivité
        - profondeur cumulée
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,5))
    else:
        fig = ax.figure

    n_layers = len(rho_layers)
    interfaces = build_layer_interfaces(thicknesses)
    z_bottom = interfaces[-1] + max(interfaces[-1]*0.3, 10.0)

    tops = np.r_[interfaces, interfaces[-1]]
    bottoms = np.r_[interfaces[1:], z_bottom]

    colors = ["#f4b183", "#bdd7ee", "#c5e0b4", "#ffe699", "#e2b9ff"]

    for i in range(n_layers):
        color = colors[i % len(colors)]
        ax.fill_betweenx([tops[i],bottoms[i]], 0, rho_layers[i],
                         color=color, alpha=0.6)

        # Étiquette de la couche
        ax.text(
            rho_layers[i]*1.05,
            (tops[i]+bottoms[i])/2,
            f"{rho_layers[i]:.1f} Ω·m",
            va="center",
            fontsize=9,
        )

    ax.invert_yaxis()                     # profondeur vers le bas
    ax.set_xlabel("Resistivity (Ω·m)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_title("Layered model")

    return fig, ax


# ==============================================================
# 3) CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================

st.set_page_config(
    page_title="1D DC VES — Schlumberger & Wenner",  # titre dans l’onglet
    page_icon="🌍",
    layout="wide",
)

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
# 4) BARRE LATÉRALE : paramètres du modèle et géométrie
# ==============================================================

with st.sidebar:

    # ---------------- AB/2 ----------------
    st.header("Géométrie (AB/2)")

    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input("AB/2 min (m)", min_value=0.1,
                                   value=5.0, step=0.1)
    with colA2:
        ab2_max = st.number_input("AB/2 max (m)", min_value=ab2_min+0.1,
                                   value=300.0, step=1.0)

    # nombre de points AB/2
    n_stations = st.slider("Nombre de stations",
                           min_value=8, max_value=60, value=25)

    st.caption(
        """
        **Schlumberger :** MN/2 = 0.1 × AB/2 (limité à 0.49 × AB/2)  
        **Wenner :** AB = 3a, MN = a
        """
    )

    st.divider()

    # ---------------- Modèle en couches ----------------
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

    # nombre de couches
    n_layers = st.slider("Nombre de couches", 3, 5, 4)

    # valeurs par défaut selon scénario
    if scenario == "Couche conductrice superficielle":
        default_rho = [5,30,80,200,300][:n_layers]
        default_thk = [3,10,40,100][:n_layers-1]
    elif scenario == "Nappe salée profonde":
        default_rho = [80,60,10,15,50][:n_layers]
        default_thk = [5,15,60,120][:n_layers-1]
    elif scenario == "Couche résistante sur substratum conducteur":
        default_rho = [20,100,15,10,8][:n_layers]
        default_thk = [10,30,80,150][:n_layers-1]
    else:
        default_rho = [10,30,15,50,100][:n_layers]
        default_thk = [2,8,60,120][:n_layers-1]

    # Résistivités de chaque couche
    st.subheader("ρ des couches (Ω·m)")
    layer_rhos = []
    for i in range(n_layers):
        layer_rhos.append(
            st.number_input(f"ρ couche {i+1}", min_value=0.1,
                            value=float(default_rho[i]), step=0.1)
        )

    # Épaisseurs
    thicknesses = []
    if n_layers>1:
        st.caption("Épaisseurs des premières couches (la dernière est un demi-espace).")
        for i in range(n_layers-1):
            thicknesses.append(
                st.number_input(f"Épaisseur couche {i+1} (m)", 
                                min_value=0.1, value=float(default_thk[i]))
            )

# Convertir en tableaux NumPy
rho_layers = np.r_[layer_rhos]
thicknesses = np.r_[thicknesses] if len(thicknesses) else np.array([])

st.divider()


# ==============================================================
# 5) Construction de AB/2, MN/2, et des surveys SimPEG
# ==============================================================

AB2 = np.geomspace(ab2_min, ab2_max, n_stations)   # logspace géométrique

MN2 = np.minimum(0.10 * AB2, 0.49*AB2)             # MN/2 pour Schlumberger

survey_schl = make_schlumberger_survey(AB2, MN2)
survey_wenn = make_wenner_survey(AB2)


# ==============================================================
# 6) Modélisation directe SimPEG (Forward)
# ==============================================================

rho_app_s, err_s = run_forward_1d(survey_schl, rho_layers, thicknesses)
rho_app_w, err_w = run_forward_1d(survey_wenn, rho_layers, thicknesses)

if err_s or err_w:
    st.error(f"Erreur Schlumberger: {err_s} / Wenner: {err_w}")
    st.stop()


# ==============================================================
# 7) Zone principale avec les onglets
# ==============================================================

tab_curves, tab_model, tab_theory = st.tabs(
    ["📈 Courbes", "🧱 Modèle & données", "📚 Théorie rapide"]
)


# --------------------------------------------------------------
# Onglet 1 : Courbes log–log
# --------------------------------------------------------------
with tab_curves:

    st.subheader("Courbes de sondage (log–log)")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.loglog(AB2, rho_app_s, "o-", label="Schlumberger ρₐ")
    ax.loglog(AB2, rho_app_w, "s--", label="Wenner ρₐ")

    ymin, ymax = compute_rho_limits(rho_app_s, rho_app_w)
    ax.set_ylim(ymin, ymax)

    build_log_ticks(ax)
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.set_xlabel("AB/2 (m)")
    ax.set_ylabel("Apparent resistivity ρₐ (Ω·m)")
    ax.set_title("Schlumberger vs Wenner — 1D VES (forward)")
    ax.legend()

    st.pyplot(fig, clear_figure=True)

    # Export CSV
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
        mime="text/csv"
    )


# --------------------------------------------------------------
# Onglet 2 : Modèle + tableau
# --------------------------------------------------------------
with tab_model:

    col1, col2 = st.columns(2)

    # Plot du modèle
    with col1:
        st.subheader("Modèle 1D")
        fig2, ax2 = plt.subplots(figsize=(4,5))
        plot_layer_model(rho_layers, thicknesses, ax=ax2)
        st.pyplot(fig2, clear_figure=True)

    # Tableau du modèle
    with col2:
        st.subheader("Tableau des couches")
        df_model = pd.DataFrame({
            "Couche": np.arange(1, len(rho_layers)+1),
            "Resistivity (Ω·m)": rho_layers,
            "Thickness (m)": [*thicknesses, np.nan],
            "Note": [""]*(len(rho_layers)-1) + ["Half-space"],
        })
        st.dataframe(df_model, use_container_width=True)


# --------------------------------------------------------------
# Onglet 3 : Théorie rapide (AFFICHAGE LaTeX PROPRE)
# --------------------------------------------------------------
with tab_theory:

    st.subheader("Rappels théoriques")

    # =========================
    # # Loi d'Ohm (scalaire)
    # =========================
    st.markdown("### Loi d’Ohm (scalaire)")
    st.latex(r"U = R\,I")

    # =========================
    # # Résistivité
    # =========================
    st.markdown("### Résistivité")
    st.latex(r"\rho = R\,\frac{A}{L}")
    st.markdown("- Unité : **Ω·m**")

    # =========================
    # # Conductivité
    # =========================
    st.markdown("### Conductivité")
    st.latex(r"\sigma = \frac{1}{\rho}")
    st.markdown("- Unité : **S/m**")

    # =========================
    # # Loi d'Ohm (vectorielle)
    # =========================
    st.markdown("### Loi d’Ohm (vectorielle)")
    st.latex(r"\vec{J} = \sigma\,\vec{E}")
    st.markdown(
        """
        - \(\\vec{J}\) : densité de courant (**A/m²**)  
        - \(\\vec{E}\) : champ électrique (**V/m**)  
        """
    )

    # =========================
    # # VES (Vertical Electrical Sounding)
    # =========================
    st.markdown("### VES (Vertical Electrical Sounding)")
    st.markdown(
        """
        - On augmente **AB/2** → la **profondeur d’investigation** augmente  
        - On mesure **ρₐ(AB/2)** pour détecter les **changements de couches**  
        """
    )

    # =========================
    # # Dispositif Schlumberger
    # =========================
    st.markdown("### Dispositif Schlumberger")
    st.markdown(
        """
        - **A, B** : électrodes de courant  
        - **M, N** : électrodes de potentiel proches du centre (**MN/2 ≪ AB/2**)  
        """
    )

    # =========================
    # # Dispositif Wenner
    # =========================
    st.markdown("### Dispositif Wenner")
    st.markdown(
        """
        - 4 électrodes régulièrement espacées de **a** : **A–M–N–B**  
        """
    )
    st.latex(r"AB = 3a \quad \text{et} \quad MN = a")

    st.info("Astuce : change le scénario dans la sidebar pour voir l’effet d’une couche conductrice ou résistante.")
