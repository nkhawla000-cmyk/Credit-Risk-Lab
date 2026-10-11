
import pandas as pd
import altair as alt

# Charger les données
df = pd.read_csv(
    "data/german.data",
    sep=r"\s+",
    header=None
)

# Préparer les variables
df = df.rename(columns={
    1: "duree_mois",
    20: "risque"
})

df["risque"] = df["risque"].map({
    1: "Bon risque",
    2: "Mauvais risque"
})

# Calculer les moyennes
moyennes = (
    df.groupby("risque", as_index=False)
    ["duree_mois"].mean()
)

# Construire le graphique
graphique = alt.Chart(moyennes).mark_bar().encode(
    x=alt.X(
        "risque:N",
        title="Catégorie de risque",
        sort=["Bon risque", "Mauvais risque"]
    ),
    y=alt.Y(
        "duree_mois:Q",
        title="Durée moyenne (mois)"
    ),
    color=alt.Color("risque:N", legend=None),
    tooltip=["risque", "duree_mois"]
).properties(
    title="Durée moyenne du crédit selon le risque",
    width=500,
    height=300
)

# Enregistrer le graphique
graphique.save("reports/eda_duree_credit.html")

print("Graphique créé avec succès !")

# Analyse de la distribution du montant du crédit

df["montant_credit"] = df[4]

graphique_montant = (
    alt.Chart(df[["risque", "montant_credit"]])
    .mark_boxplot()
    .encode(
        x=alt.X(
            "risque:N",
            title="Catégorie de risque",
            sort=["Bon risque", "Mauvais risque"]
        ),
        y=alt.Y(
            "montant_credit:Q",
            title="Montant du crédit"
        ),
        color=alt.Color("risque:N", legend=None)
    )
    .properties(
        title="Distribution du montant du crédit selon le risque",
        width=500,
        height=350
    )
)

graphique_montant.save(
    "reports/eda_montant_credit.html"
)

print("Graphique du montant créé avec succès !")
