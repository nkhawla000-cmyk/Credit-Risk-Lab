
import pandas as pd

# 1. Charger les données
df = pd.read_csv(
    "data/german.data",
    sep=r"\s+",
    header=None
)

# 2. Nommer les variables étudiées
df = df.rename(columns={
    1: "duree_mois",
    4: "montant_credit",
    12: "age",
    20: "risque"
})

# 3. Identifier les catégories de risque
df["risque"] = df["risque"].map({
    1: "Bon risque",
    2: "Mauvais risque"
})

# 4. Afficher les statistiques descriptives
print("\nSTATISTIQUES DESCRIPTIVES")
print(df[["duree_mois", "montant_credit", "age"]].describe())

# 5. Comparer les deux catégories de clients
print("\nMOYENNES PAR CATEGORIE DE RISQUE")
print(
    df.groupby("risque")[
        ["duree_mois", "montant_credit", "age"]
    ].mean().round(2)
)

# Analyse détaillée du montant du crédit

print("\nMEDIANES ET QUARTILES PAR RISQUE")

statistiques = df.groupby("risque")["montant_credit"].agg(
    effectif="count",
    moyenne="mean",
    mediane="median",
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75),
    ecart_type="std"
)

statistiques["IQR"] = (
    statistiques["Q3"] - statistiques["Q1"]
)

print(statistiques.round(2).to_string())

