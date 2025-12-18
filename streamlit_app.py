# ==============================================================
# STYLE + INTRODUCTION
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

st.markdown(
    """
    Simulation de **sondage électrique vertical (VES)** en milieu **1D**.  
    L’application calcule la **résistivité apparente** ρₐ en fonction de **AB/2**
    pour les dispositifs **Schlumberger** et **Wenner**.

    Cette application est conçue comme un **outil pédagogique** permettant
    de relier la **théorie du courant continu** aux **courbes de sondage électrique**.
    """
)

st.divider()
