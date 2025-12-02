# ==============================================================
# 1D DC Forward Modelling (SimPEG) — Schlumberger + Wenner
# Version "pédagogique" optimisée + ultra commentée
#
# Objectif :
#  - Définir un modèle 1D en couches (ρ et épaisseurs)
#  - Construire les géométries Schlumberger et Wenner à partir de AB/2
#  - Calculer les résistivités apparentes ρa avec SimPEG
#  - Afficher les courbes de sondage + le modèle de couches
# ==============================================================

# -----------------------------
# 0) IMPORTS DES LIBRAIRIES
# -----------------------------
import numpy as np              # calcul numérique, tableaux
import pandas as pd             # tableaux de données (pour export CSV + tableau de modèle)
import matplotlib.pyplot as plt # graphiques
import streamlit as st          # interface web Streamlit

# SimPEG : module DC résistivité + "maps" pour relier le modèle aux paramètres physiques
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

# Outils pour axes logarithmiques jolis
from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter


# ==============================================================
# 1) FONCTIONS UTILITAIRES
# ==============================================================

def build_log_ticks(ax):
    """
    # Configure les axes x et y en échelle log avec des graduations propres.
    # ax : objet Axes de matplotlib.
    """
    # Graduation principale : 1, 10, 100, ...
    major_locator = LogLocator(base=10.0, subs=(1.0,))
    # Graduation secondaire : 2,3,...9 entre chaque décennie
    minor_locator = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)

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


def compute_rho_limits(*arrays):
    """
    # Calcule des bornes min / max propres pour l’axe Y en log.
    # Prend plusieurs tableaux (ρa Schlumberger, ρa Wenner) et renvoie (ymin, ymax).
    """
    # On prend le min global et le max global de tous les tableaux passés
    vals = np.hstack(arrays)
    ymin = vals.min()
    ymax = vals.max()

    # On arrondit aux puissances de 10 entières (pour un axe log propre)
    ymin = 10 ** np.floor(np.log10(ymin))
    ymax = 10 ** np.ceil(np.log10(ymax))
    return ymin, ymax


def build_layer_interfaces(thicknesses):
    """
    # À partir des épaisseurs des couches supérieures, construit
    # les profondeurs d'interface (z) de chaque couche.
    #
    # thicknesses : tableau des épaisseurs des N-1 premières couches.
    # Retourne : tableau des profondeurs d’interface (0, z1, z2, ..., zN-1)
    """
    if len(thicknesses) == 0:
        # Cas d’une seule couche (demi-espace)
        return np.array([0.0])
    return np.r_[0.0, np.cumsum(thicknesses)]


def make_schlumberger_survey(AB2, MN2):
    """
    # Construit le "survey" SimPEG pour un dispositif Schlumberger 1D.
    #
    # AB2 : tableau des AB/2 (demi-distance entre A et B)
    # MN2 : tableau des MN/2 (demi-distance entre M et N)
    #
    # Retour : objet dc.Survey
    """
    src_list = []
    eps = 1e-6  # petit décalage pour éviter M=N exactement (problème num.)

    for L, a in zip(AB2, MN2):
        # Électrodes de courant A et B aux positions -L et +L
        A = np.r_[-L, 0.0, 0.0]
        B = np.r_[+L, 0.0, 0.0]

        # Électrodes de potentiel M et N proches du centre
        M = np.r_[-(a - eps), 0.0, 0.0]
        N = np.r_[+(a - eps), 0.0, 0.0]

        # Récepteur : dipôle MN, on demande la résistivité apparente directement
        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")

        # Source : dipôle de courant AB associé au récepteur
        src = dc.sources.Dipole([rx], A, B)
        src_list.append(src)

    return dc.Survey(src_list)


def make_wenner_survey(AB2):
    """
    # Construit le "survey" SimPEG pour un dispositif Wenner 1D.
    #
    # Wenner : A–M–N–B régulièrement espacés de 'a'.
    # On a AB = 3a, donc AB/2 = 1.5a ⇒ a = (2/3) * (AB/2).
    #
    # AB2 : tableau de AB/2 (m)
    #
    # Retour : objet dc.Survey
    """
    src_list = []

    for L in AB2:
        a = (2.0 / 3.0) * L  # pas Wenner

        # Positions symétriques autour de 0
        A = np.r_[-1.5 * a, 0.0, 0.0]
        M = np.r_[-0.5 * a, 0.0, 0.0]
        N = np.r_[+0.5 * a, 0.0, 0.0]
        B = np.r_[+1.5 * a, 0.0, 0.0]

        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
        src = dc.sources.Dipole([rx], A, B)
        src_list.append(src)

    return dc.Survey(src_list)


def run_forward_1d(survey, rho_layers, thicknesses):
    """
    # Lance la modélisation 1D avec SimPEG pour un survey donné.
    #
    # survey      : objet dc.Survey (Schlumberger ou Wenner)
    # rho_layers  : tableau des résistivités de chaque couche (Ω·m)
    # thicknesses : épaisseurs des N-1 premières couches (m)
    #
    # Retour :
    #   - data : ρa calculée (tableau)
    #   - err  : message d'erreur ou None si tout va bien
    """
    rho_map = maps.IdentityMap(nP=len(rho_layers))  # map identité : modèle = ρ

    sim = dc.simulation_1d.Simulation1DLayers(
        survey=survey,
        rhoMap=rho_map,
        thicknesses=thicknesses,
    )

    try:
        data = sim.dpred(rho_layers)  # prédiction des données (ici ρa)
        return data, None
    except Exception as e:
        # On renvoie None + message d'erreur
        return None, str(e)


def plot_layer_model(rho_layers, thicknesses, ax=None):
    """
    # Trace le modèle en couches (ρ en x, profondeur en y).
    #
    # rho_layers  : résistivités des couches (Ω·m)
    # thicknesses : épaisseurs des N-1 premières couches (m)
    # ax          : axe matplotlib, ou None pour en créer un.
    #
    # Retour : (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig = ax.figure

    n_layers = len(rho_layers)

    # Interfaces en profondeur (0, z1, z2, ..., zN-1)
    interfaces = build_layer_interfaces(thicknesses)

    # Profondeur du bas de la dernière couche (on descend un peu plus)
    z_bottom = interfaces[-1] + max(interfaces[-1] * 0.3, 10.0)

    # Tops et bottoms de chaque bloc
    tops = np.r_[interfaces, interfaces[-1]]
    bottoms = np.r_[interfaces[1:], z_bottom]

    for i in range(n_layers):
        # Remplissage horizontal pour chaque couche
        ax.fill_betweenx([tops[i], bottoms[i]], 0, rho_layers[i], alpha=0.35)
        # Étiquette de résistivité
        ax.text(
            rho_layers[i] * 1.05,
            (tops[i] + bottoms[i]) / 2,
            f"{rho_layers[i]:.1f} Ω·m",
            va="center",
            fontsize=9,
        )

    ax.invert_yaxis()  # profondeur positive vers le bas
    ax.set_xlabel("Resistivity (Ω·m)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, ls=":")
    ax.set_title("Layered model")

    return fig, ax


# ==============================================================
# 2) CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================

st.set_page_config(
    page_title="1D DC Forward (SimPEG)",
    page_icon="🪪",
    layout="wide"
)

st.title("1D DC Resistivity — Forward Modelling (Schlumberger vs Wenner)")

st.markdown(
    """
    Configure un modèle en couches et une gamme de **AB/2**,
    puis calcule les courbes de **résistivité apparente** pour les dispositifs
    **Schlumberger** et **Wenner** (1D).  
    Basé sur `simpeg.electromagnetics.static.resistivity.simulation_1d.Simulation1DLayers`.
    """
)

st.divider()

# ==============================================================
# 3) BARRE LATERALE : GEOMETRIE + MODELE DE COUCHES
# ==============================================================

with st.sidebar:
    st.header("Géométrie (AB/2)")

    # --- Choix de AB/2 min / max ---
    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input(
            "AB/2 min (m)",
            min_value=0.1,
            value=5.0,
            step=0.1,
            format="%.2f",
            help="Plus petit demi-écartement des électrodes de courant."
        )
    with colA2:
        ab2_max = st.number_input(
            "AB/2 max (m)",
            min_value=ab2_min + 0.1,
            value=300.0,
            step=1.0,
            format="%.2f",
            help="Plus grand demi-écartement des électrodes de courant."
        )

    # --- Nombre de mesures (points AB/2) ---
    n_stations = st.slider(
        "Nombre de stations",
        min_value=8,
        max_value=60,
        value=25,
        step=1,
        help="Nombre de valeurs AB/2 entre min et max (échelle géométrique)."
    )

    st.caption(
        """
        **Schlumberger :** MN/2 est fixé à 10 % de AB/2 (et limité à 0,49·AB/2).  
        **Wenner :** AB = 3a, MN = a, centré en x = 0.
        """
    )

    st.divider()
    st.header("Modèle de couches")

    # --- Nombre de couches ---
    n_layers = st.slider(
        "Nombre de couches",
        min_value=3,
        max_value=5,
        value=4,
        help="La dernière couche est un demi-espace (épaisseur infinie)."
    )

    # Valeurs par défaut "raisonnables" pour ρ et épaisseurs
    default_rho = [10.0, 30.0, 15.0, 50.0, 100.0][:n_layers]
    default_thk = [2.0, 8.0, 60.0, 120.0][:max(0, n_layers - 1)]

    # --- Résistivités des couches ---
    layer_rhos = []
    st.subheader("Résistivité des couches (Ω·m)")
    for i in range(n_layers):
        rho_i = st.number_input(
            f"ρ couche {i + 1} (Ω·m)",
            min_value=0.1,
            value=float(default_rho[i]),
            step=0.1,
        )
        layer_rhos.append(rho_i)

    # --- Épaisseurs des N-1 couches supérieures ---
    thicknesses = []
    if n_layers > 1:
        st.caption(
            "Épaisseurs pour les **N−1 premières couches** "
            "(la dernière est un demi-espace)."
        )
        for i in range(n_layers - 1):
            thk_i = st.number_input(
                f"Épaisseur couche {i + 1} (m)",
                min_value=0.1,
                value=float(default_thk[i]),
                step=0.1,
            )
            thicknesses.append(thk_i)

# Conversion des listes en tableaux NumPy
rho_layers = np.r_[layer_rhos]
thicknesses = np.r_[thicknesses] if len(thicknesses) else np.array([])

st.divider()

# ==============================================================
# 4) CONSTRUCTION DES GEOMETRIES (AB/2, Schlumberger, Wenner)
# ==============================================================

# --- AB/2 échantillonné de façon géométrique (log) ---
AB2 = np.geomspace(ab2_min, ab2_max, n_stations)

# --- Schlumberger : MN/2 = 0.1 * AB/2, limité à 0.49*AB/2 ---
MN2 = np.minimum(0.10 * AB2, 0.49 * AB2)

# Création des surveys SimPEG
survey_schl = make_schlumberger_survey(AB2, MN2)
survey_wenn = make_wenner_survey(AB2)

# ==============================================================
# 5) MODELLISATION DIRECTE (FORWARD) AVEC SIMPEG
# ==============================================================

rho_app_s, err_s = run_forward_1d(survey_schl, rho_layers, thicknesses)
rho_app_w, err_w = run_forward_1d(survey_wenn, rho_layers, thicknesses)

# On vérifie s’il y a eu des erreurs
if err_s or err_w:
    st.error(
        "La modélisation directe a échoué :\n"
        f"- Schlumberger : {err_s}\n"
        f"- Wenner : {err_w}"
    )
    st.stop()  # on arrête l’app ici pour éviter les plantages plus loin


# ==============================================================
# 6) AFFICHAGE DES RESULTATS
# ==============================================================

col_curves, col_model = st.columns([2, 1])

# -----------------------------------
# 6.1 Courbes de résistivité apparente
# -----------------------------------
with col_curves:
    st.subheader("Courbes de sondage (log–log)")

    fig, ax = plt.subplots(figsize=(7, 5))

    # Tracé des deux dispositifs
    ax.loglog(AB2, rho_app_s, "o-", label="Schlumberger ρₐ")
    ax.loglog(AB2, rho_app_w, "s--", label="Wenner ρₐ")

    # Limites Y propres (puissances de 10 entières)
    ymin, ymax = compute_rho_limits(rho_app_s, rho_app_w)
    ax.set_ylim(ymin, ymax)

    # Axes log bien formatés
    build_log_ticks(ax)

    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.set_xlabel("AB/2 (m)")
    ax.set_ylabel("Apparent resistivity ρₐ (Ω·m)")
    ax.set_title("Schlumberger vs Wenner — 1D VES (forward)")
    ax.legend()

    st.pyplot(fig, clear_figure=True)

    # --- Export CSV des données synthétiques ---
    df_out = pd.DataFrame(
        {
            "AB/2 (m)": AB2,
            "MN/2 Schlumberger (m)": MN2,
            "ρa Schlumberger (Ω·m)": rho_app_s,
            "ρa Wenner (Ω·m)": rho_app_w,
        }
    )

    st.download_button(
        "⬇️ Télécharger les données synthétiques (CSV)",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_VES_Schlumberger_Wenner.csv",
        mime="text/csv",
    )

# ----------------------
# 6.2 Modèle de couches
# ----------------------
with col_model:
    st.subheader("Modèle 1D en couches")

    fig2, ax2 = plt.subplots(figsize=(4, 5))
    plot_layer_model(rho_layers, thicknesses, ax=ax2)
    st.pyplot(fig2, clear_figure=True)

    # Tableau récapitulatif des couches
    model_df = pd.DataFrame(
        {
            "Couche": np.arange(1, len(rho_layers) + 1),
            "Resistivity (Ω·m)": rho_layers,
            "Thickness (m)": [*thicknesses, np.nan],
            "Note": [""] * (len(rho_layers) - 1) + ["Half-space"],
        }
    )
    st.dataframe(model_df, use_container_width=True)

# ==============================================================
# 7) NOTE PEDAGOGIQUE
# ==============================================================

st.caption(
    """
    Notes :
    - En Schlumberger, MN/2 est fixé à 10 % de AB/2 (et limité à 0,49·AB/2)
      pour éviter les problèmes numériques et le chevauchement des électrodes.
    - En Wenner, AB = 3a et MN = a, l’ensemble est centré à x = 0.
    - Si des instabilités apparaissent pour des géométries extrêmes, réduis
      la gamme AB/2 ou le nombre de stations.
    """
)
