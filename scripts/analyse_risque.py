
import pandas as pd
import altair as alt

# Charger les données
df = pd.read_csv(
    "data/german.data",
    sep=r"\s+",
    header=None
)

# Variables étudiées
df["montant_credit"] = df[4]
df["defaut"] = (df[20] == 2).astype(int)

# Répartition en cinq groupes de montants
df["groupe_montant"] = pd.qcut(
    df["montant_credit"],
    q=5,
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)

# Calculer le taux de mauvais risque
resultats = (
    df.groupby("groupe_montant", observed=True)
    .agg(
        effectif=("defaut", "size"),
        taux_mauvais_risque=("defaut", "mean")
    )
    .reset_index()
)

resultats["taux_mauvais_risque"] *= 100

print("\nTAUX DE MAUVAIS RISQUE PAR QUINTILE")
print(resultats.round(2).to_string(index=False))

# Graphique
graphique = alt.Chart(resultats).mark_bar().encode(
    x=alt.X(
        "groupe_montant:N",
        title="Quintile du montant du crédit",
        sort=["Q1", "Q2", "Q3", "Q4", "Q5"]
    ),
    y=alt.Y(
        "taux_mauvais_risque:Q",
        title="Taux de mauvais risque (%)",
        scale=alt.Scale(domain=[0, 100])
    ),
    tooltip=[
        "groupe_montant",
        "effectif",
        alt.Tooltip(
            "taux_mauvais_risque:Q",
            format=".2f"
        )
    ]
).properties(
    title="Taux de mauvais risque selon le montant",
    width=550,
    height=350
)

graphique.save("reports/taux_mauvais_risque.html")

print("\nGraphique enregistré avec succès !")

from scipy.stats import chi2_contingency

# Tableau de contingence
tableau = pd.crosstab(
    df["groupe_montant"],
    df["defaut"]
)

print("\nTABLEAU DE CONTINGENCE")
print(tableau)

# Test statistique du Khi-deux
chi2, p_value, ddl, attendus = chi2_contingency(tableau)

print("\nRESULTATS DU TEST DU KHI-DEUX")
print(f"Statistique : {chi2:.4f}")
print(f"Degrés de liberté : {ddl}")
print(f"P-value : {p_value:.6f}")

if p_value < 0.05:
    print("Rejet de H0 : association statistiquement significative")
else:
    print("H0 non rejetée au seuil de 5 %")
