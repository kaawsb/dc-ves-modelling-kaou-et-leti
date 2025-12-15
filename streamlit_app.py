with tab_theory:

    st.subheader("Rappels théoriques")

    # ------------------------------
    # Loi d’Ohm (scalaire)
    # ------------------------------
    st.markdown("### Loi d’Ohm (scalaire)")
    st.latex(r"U = R\,I")

    # ------------------------------
    # Résistivité
    # ------------------------------
    st.markdown("### Résistivité")
    st.latex(r"\rho = R\,\frac{A}{L}")
    st.markdown("- Unité : **Ω·m**")

    # ------------------------------
    # Conductivité
    # ------------------------------
    st.markdown("### Conductivité")
    st.latex(r"\sigma = \frac{1}{\rho}")
    st.markdown("- Unité : **S/m**")

    # ------------------------------
    # Loi d’Ohm vectorielle
    # ------------------------------
    st.markdown("### Loi d’Ohm (vectorielle)")
    st.latex(r"\vec{J} = \sigma\,\vec{E}")
    st.markdown(
        """
        - **J** : densité de courant (A/m²)  
        - **E** : champ électrique (V/m)
        """
    )

    # ------------------------------
    # VES
    # ------------------------------
    st.markdown("### VES (Vertical Electrical Sounding)")
    st.markdown(
        """
        - Quand **AB/2 augmente**, la **profondeur d’investigation augmente**  
        - On mesure la **résistivité apparente ρₐ(AB/2)**  
        """
    )

    # ------------------------------
    # Dispositifs
    # ------------------------------
    st.markdown("### Dispositif Schlumber
