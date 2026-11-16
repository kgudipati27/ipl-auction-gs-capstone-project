"""
========================================================================================
Project Title: When to Pick an Overseas Player and When to Pick an Indian Player in
               the IPL Auction: A Data Analysis Project
========================================================================================
Problem Statement:
In the Indian Premier League (IPL), every team can have a maximum of 8 overseas players
in the squad and only 4 overseas players in the playing 11. Because of this rule, teams
must carefully decide whether to spend money on an overseas player or an Indian player
during the auction.
This project aims to study past IPL auction data, player prices, performance statistics,
and team results to find clear patterns. The goal is to understand in which situations
it is better to pick an overseas player and in which situations an Indian player is the
smarter choice.
========================================================================================

Analysis Context: Ahead of IPL 2024 Mega-Auction (using 2018-2023 historical data).
Methods Used (Strictly from Curriculum & Selected Skills):
1. DBMS & Pandas: Merging auction, delivery, and match datasets; data cleaning.
2. MCDM (Saaty's AHP Scale + TOPSIS): Objective player performance scoring.
3. Data Mining (K-Means Clustering): Grouping players into Cost-vs-Value tiers.
4. Machine Learning (One-Hot Encoding, Train-Test Split, Multiple Linear Regression):
   Measuring the Overseas Price Premium & predicting fair auction value.
5. Win Prediction Model (Logistic Regression, Decision Tree, KNN):
   Proving how Overseas vs Indian contributions affect match wins.
6. Accounting & Financial Management + PuLP Linear Programming:
   Opportunity cost, squad constraints, and break-even max bid price.
========================================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, silhouette_score
)

os.makedirs("outputs", exist_ok=True)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E0"

# -----------------------------------------------------------------------------
# STEP 1: DATA LOADING, CLEANING & JOINING (DBMS + DATA MINING PREPROCESSING)
# -----------------------------------------------------------------------------
print("=" * 75)
print("STEP 1: DATA LOADING, CLEANING & ENTITY RESOLUTION")
print("=" * 75)

auc = pd.read_csv("ipl_auctions_2018_2026.csv")
deliv = pd.read_csv("IPL Deliveries Data.csv", low_memory=False)
matches = pd.read_csv("IPL Matches Data.csv")

# Clean role typo ("Bowler Allrounder" -> "Allrounder")
auc["Player Role"] = auc["Player Role"].replace({"Bowler Allrounder": "Allrounder"})
auc["Is_Sold"] = (auc["Status"] != "Unsold").astype(int)
auc["Is_OS"] = (auc["Player Nationality"] != "India").astype(int)
auc["Is_Capped"] = (auc["Capped / Uncapped"] == "Capped").astype(int)
auc["Is_Retained"] = (~auc["Status"].isin(["Sold", "Unsold"])).astype(int)

# Deduplicate (Season, Player Name) keeping Sold entry first (e.g. Shashank Singh 2024)
auc = auc.sort_values(["Season", "Player Name", "Is_Sold"], ascending=[True, True, False])
auc = auc.drop_duplicates(subset=["Season", "Player Name"], keep="first").reset_index(drop=True)

# Handle structural missing prices: Retained lack Base Price; Unsold lack Auction Price
auc["Base_Clean"] = auc["Base Price (Lakhs)"].fillna(20.0)
auc["Eval_Price_Lakhs"] = auc["Auction / Retention Price (Lakhs)"].fillna(auc["Base_Clean"])

# Map full names in auction CSV to abbreviated names in deliveries CSV
deliv_players = sorted(set(deliv["Striker"].dropna()).union(set(deliv["Bowler"].dropna())))
by_last, by_exact_lower = {}, {}
for p in deliv_players:
    pc = p.strip()
    by_exact_lower[pc.lower()] = pc
    parts = pc.split()
    by_last.setdefault((parts[-1].lower(), parts[0][0].lower()), []).append(pc)
    by_last.setdefault((parts[-1].lower(), ""), []).append(pc)

ALIASES = {
    "dinesh karthik": "KD Karthik", "lokesh rahul": "KL Rahul",
    "prasidh krishna": "M Prasidh Krishna", "varun chakravarthy": "CV Varun",
    "varun chakaravarthy": "CV Varun", "srikar bharat": "KS Bharat",
    "dushmantha chameera": "PVD Chameera", "lasith malinga": "SL Malinga",
    "sai sudharsan": "B Sai Sudharsan", "wanindu hasaranga": "PWH de Silva",
    "bhanuka rajapaksa": "PBB Rajapaksa", "hardus viljoen": "GC Viljoen",
    "gurkeerat singh mann": "Gurkeerat Singh", "suryakumar yadav": "SA Yadav",
    "shimron hetmyer": "SO Hetmyer", "yashasvi jaiswal": "YBK Jaiswal",
    "ruturaj gaikwad": "RD Gaikwad", "quinton de kock": "Q de Kock",
    "faf du plessis": "F du Plessis", "heinrich klaasen": "H Klaasen",
    "nicholas pooran": "N Pooran", "trent boult": "TA Boult",
    "kagiso rabada": "K Rabada", "pat cummins": "PJ Cummins",
    "mitchell starc": "MA Starc", "jasprit bumrah": "JJ Bumrah",
    "yuzvendra chahal": "YS Chahal", "ravindra jadeja": "RA Jadeja",
    "hardik pandya": "HH Pandya", "sanju samson": "SV Samson",
    "rishabh pant": "RR Pant", "virat kohli": "V Kohli",
    "rohit sharma": "RG Sharma", "jos buttler": "JC Buttler",
    "sunil narine": "SP Narine", "andre russell": "AD Russell",
    "glenn maxwell": "GJ Maxwell", "jason holder": "JO Holder",
    "matheesha pathirana": "M Pathirana", "maheesh theekshana": "M Theekshana",
    "lockie ferguson": "LH Ferguson", "alzarri joseph": "AS Joseph",
    "josh hazlewood": "JR Hazlewood", "marcus stoinis": "MP Stoinis",
    "sam curran": "SM Curran", "liam livingstone": "LS Livingstone",
    "moeen ali": "MM Ali", "ravichandran ashwin": "R Ashwin",
    "axar patel": "AR Patel", "bhuvneshwar kumar": "B Kumar",
    "harshal patel": "HV Patel", "ab de villiers": "AB de Villiers",
    "chris gayle": "CH Gayle", "kieron pollard": "KA Pollard",
    "david warner": "DA Warner", "shane watson": "SR Watson",
    "dwayne bravo": "DJ Bravo", "kane williamson": "KS Williamson",
    "jonny bairstow": "JM Bairstow", "anrich nortje": "A Nortje",
    "daryl mitchell": "DJ Mitchell", "rilee rossouw": "RR Rossouw",
    "travis head": "TM Head", "rinku singh": "RK Singh",
    "cameron green": "C Green", "chris green": "CJ Green",
    "jitesh sharma": "JM Sharma", "karn sharma": "KV Sharma",
    "prabhsimran singh": "P Simran Singh", "shahrukh khan": "M Shahrukh Khan",
    "mujeeb ur rahman": "Mujeeb Ur Rahman", "mustafizur rahman": "Mustafizur Rahman",
    "kuldeep yadav": "Kuldeep Yadav", "ravi bishnoi": "Ravi Bishnoi",
    "avesh khan": "Avesh Khan", "arshdeep singh": "Arshdeep Singh",
    "abhishek sharma": "Abhishek Sharma", "ashutosh sharma": "Ashutosh Sharma",
    "mukesh kumar": "Mukesh Kumar", "mohsin khan": "Mohsin Khan",
    "shashank singh": "Shashank Singh", "ramandeep singh": "Ramandeep Singh",
    "simarjeet singh": "Simarjeet Singh", "swapnil singh": "Swapnil Singh",
    "suyash sharma": "Suyash Sharma", "sandeep sharma": "Sandeep Sharma",
    "shahbaz ahmed": "Shahbaz Ahmed", "navdeep saini": "Navdeep Saini",
    "fabian allen": "FA Allen", "finn allen": "FH Allen"
}


def match_player_name(full_name):
    fn = str(full_name).strip()
    if fn.lower() in ALIASES:
        return ALIASES[fn.lower()]
    if fn.lower() in by_exact_lower:
        return by_exact_lower[fn.lower()]
    parts = re.sub(r"[^a-zA-Z\-\s]", "", fn).split()
    if not parts:
        return None
    cands = by_last.get((parts[-1].lower(), parts[0][0].lower()), [])
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        for c in cands:
            if c.split()[0].lower() == parts[0].lower():
                return c
        return None
    for c in by_last.get((parts[-1].lower(), ""), []):
        if any(p[0].lower() == c.split()[0][0].lower() for p in parts[:-1]):
            return c
    return None


auc["Deliv_Name"] = auc["Player Name"].apply(match_player_name)

# Prepare delivery flags
deliv["Is_Wicket"] = deliv["Wicket Type"].isin(
    ["bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"]
).astype(int)
deliv["Runs_Conceded"] = deliv["Runs Off Bat"] + deliv["Extras"]

# Filter pre-2024 period (2018-2023)
a18_23 = auc[(auc["Season"] >= 2018) & (auc["Season"] <= 2023)].copy()
d18_23 = deliv[(deliv["Season"] >= 2018) & (deliv["Season"] <= 2023)].copy()
print(f"2018-2023 Auction Entries: {len(a18_23)} | Acquired: {a18_23['Is_Sold'].sum()} | Unsold: {(a18_23['Is_Sold']==0).sum()}")

# -----------------------------------------------------------------------------
# STEP 2: MARKET PRICE REALITY & VALUE EFFICIENCY (2018-2023)
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print("STEP 2: AUCTION MARKET PRICES, UNSOLD RISK & VALUE PER CRORE")
print("=" * 75)

sold_only = a18_23[a18_23["Status"] == "Sold"].copy()
acq_18_23 = a18_23[a18_23["Is_Sold"] == 1].copy()

print("\nOpen-Auction ('Sold' only) Median Price in Lakhs:")
med_price = sold_only.groupby(["Player Role", "Is_OS"])["Eval_Price_Lakhs"].median().unstack()
med_price.columns = ["Indian_Median_L", "Overseas_Median_L"]
print(med_price.round(1))

print("\nUnsold Rate (%) by Role & Origin:")
unsold_rate = (1 - a18_23.groupby(["Player Role", "Is_OS"])["Is_Sold"].mean()).unstack() * 100
unsold_rate.columns = ["Indian_Unsold_%", "Overseas_Unsold_%"]
print(unsold_rate.round(1))

# Join 2018-2023 acquired players with same-season on-field output for Cost Efficiency
bat_s = d18_23.groupby(["Season", "Striker"]).agg(
    Bat_Runs=("Runs Off Bat", "sum"), Bat_Balls=("Ball", "count")
).reset_index().rename(columns={"Striker": "Deliv_Name"})

bowl_s = d18_23.groupby(["Season", "Bowler"]).agg(
    Bowl_Balls=("Ball", "count"), Bowl_Runs=("Runs_Conceded", "sum"), Bowl_Wkts=("Is_Wicket", "sum")
).reset_index().rename(columns={"Bowler": "Deliv_Name"})

perf_18_23 = acq_18_23.merge(bat_s, on=["Season", "Deliv_Name"], how="left") \
                      .merge(bowl_s, on=["Season", "Deliv_Name"], how="left").fillna(0)
perf_18_23["Price_Cr"] = perf_18_23["Eval_Price_Lakhs"] / 100.0
perf_18_23["Played"] = ((perf_18_23["Bat_Balls"] + perf_18_23["Bowl_Balls"]) > 0).astype(int)
played_df = perf_18_23[perf_18_23["Played"] == 1].copy()
played_df["Runs_per_Cr"] = played_df["Bat_Runs"] / played_df["Price_Cr"].clip(lower=0.2)
played_df["Wkts_per_Cr"] = played_df["Bowl_Wkts"] / played_df["Price_Cr"].clip(lower=0.2)

print("\nMean Value Efficiency (Played Players 2018-2023):")
print(played_df.groupby(["Player Role", "Is_OS"])[["Price_Cr", "Bat_Runs", "Bowl_Wkts", "Runs_per_Cr", "Wkts_per_Cr"]].mean().round(2))

# Plot Figure 1: Open Auction Median Price & Value Efficiency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.patch.set_facecolor("#F8FAFC")

roles = ["Batter", "Allrounder", "Bowler", "WK-Batter"]
x_pos = np.arange(len(roles))
w_bar = 0.35

ind_meds = [med_price.loc[r, "Indian_Median_L"] for r in roles]
os_meds = [med_price.loc[r, "Overseas_Median_L"] for r in roles]
b1 = ax1.bar(x_pos - w_bar/2, ind_meds, w_bar, label="Indian (Open Auction)", color="#2B6CB0", edgecolor="black")
b2 = ax1.bar(x_pos + w_bar/2, os_meds, w_bar, label="Overseas (Open Auction)", color="#DD6B20", edgecolor="black")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(roles, fontweight="bold")
ax1.set_ylim(0, 270)
ax1.set_ylabel("Median Sold Price (Rs. Lakhs)", fontweight="bold")
ax1.set_title("1A. Open Auction Median Price (2018-2023)", fontweight="bold")
ax1.legend()
ax1.grid(axis="y", linestyle=":", alpha=0.6)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax1.annotate(f"{h:.0f}L", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

eff_summary = played_df.groupby(["Player Role", "Is_OS"])["Runs_per_Cr"].mean().unstack()
ind_rpc = [eff_summary.loc[r, 0] for r in ["Batter", "WK-Batter", "Allrounder"]]
os_rpc = [eff_summary.loc[r, 1] for r in ["Batter", "WK-Batter", "Allrounder"]]
x_eff = np.arange(3)
b3 = ax2.bar(x_eff - w_bar/2, ind_rpc, w_bar, label="Indian (Runs per Cr)", color="#2B6CB0", edgecolor="black")
b4 = ax2.bar(x_eff + w_bar/2, os_rpc, w_bar, label="Overseas (Runs per Cr)", color="#DD6B20", edgecolor="black")
ax2.set_xticks(x_eff)
ax2.set_xticklabels(["Batter", "WK-Batter", "Allrounder"], fontweight="bold")
ax2.set_ylabel("Mean Runs per Rs. Crore", fontweight="bold")
ax2.set_title("1B. Batting Value Efficiency (Runs / Crore)", fontweight="bold")
ax2.legend()
ax2.grid(axis="y", linestyle=":", alpha=0.6)
for bar in list(b3) + list(b4):
    h = bar.get_height()
    ax2.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/fig1_market_prices_and_efficiency.png", dpi=200, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# STEP 3: MCDM (SAATY AHP + TOPSIS), K-MEANS CLUSTERING & PRICE REGRESSION
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print("STEP 3: SAATY AHP + TOPSIS, K-MEANS CLUSTERING & PRICE REGRESSION")
print("=" * 75)

# Saaty's 1-9 Pairwise Comparison Matrix for [Bat_Runs, Bat_SR, Death_SR, Bowl_Wkts, Econ_Save]
A_saaty = np.array([
    [1,   2,   2,   1,   2],
    [0.5, 1,   1,   0.5, 1],
    [0.5, 1,   1,   0.5, 1],
    [1,   2,   2,   1,   2],
    [0.5, 1,   1,   0.5, 1]
], dtype=float)
eigvals, eigvecs = np.linalg.eig(A_saaty)
idx_max = np.argmax(eigvals.real)
ahp_weights = eigvecs[:, idx_max].real
ahp_weights = ahp_weights / ahp_weights.sum()
lambda_max = eigvals[idx_max].real
CR = ((lambda_max - 5) / 4) / 1.12
print("Saaty AHP Weights [Runs, SR, Death_SR, Wkts, EconSave]:", np.round(ahp_weights, 3), "| Consistency Ratio (CR):", round(CR, 4))


def compute_pre_auction_topsis(target_year, lookback=5):
    """Computes TOPSIS score (0-100) using ONLY deliveries strictly BEFORE target_year."""
    hd = deliv[(deliv["Season"] >= target_year - lookback) & (deliv["Season"] < target_year)]
    bat = hd.groupby("Striker").agg(Bat_Runs=("Runs Off Bat", "sum"), Bat_Balls=("Ball", "count")).reset_index().rename(columns={"Striker": "Deliv_Name"})
    dbat = hd[hd["Ball"] >= 15.0].groupby("Striker").agg(D_Runs=("Runs Off Bat", "sum"), D_Balls=("Ball", "count")).reset_index().rename(columns={"Striker": "Deliv_Name"})
    bowl = hd.groupby("Bowler").agg(Bowl_Balls=("Ball", "count"), Bowl_Runs=("Runs_Conceded", "sum"), Bowl_Wkts=("Is_Wicket", "sum")).reset_index().rename(columns={"Bowler": "Deliv_Name"})

    sub = auc[auc["Season"] == target_year].copy().reset_index(drop=True)
    sub = sub.merge(bat, on="Deliv_Name", how="left").merge(dbat, on="Deliv_Name", how="left").merge(bowl, on="Deliv_Name", how="left").fillna(0)
    sub["Bat_SR"] = np.where(sub["Bat_Balls"] >= 30, sub["Bat_Runs"] / sub["Bat_Balls"] * 100, 0)
    sub["Death_SR"] = np.where(sub["D_Balls"] >= 15, sub["D_Runs"] / sub["D_Balls"] * 100, 0)
    sub["Econ_Save"] = np.where(sub["Bowl_Balls"] >= 30, np.maximum(0, 12.0 - sub["Bowl_Runs"] / (sub["Bowl_Balls"] / 6.0)), 0)

    V = sub[["Bat_Runs", "Bat_SR", "Death_SR", "Bowl_Wkts", "Econ_Save"]].values.astype(float)
    norm = np.sqrt((V ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    Vn = (V / norm) * ahp_weights
    d_best = np.sqrt(((Vn - Vn.max(axis=0)) ** 2).sum(axis=1))
    d_worst = np.sqrt(((Vn - Vn.min(axis=0)) ** 2).sum(axis=1))
    sub["TOPSIS"] = np.round((d_worst / (d_best + d_worst + 1e-9)) * 100, 2)

    # Prior baseline for players without prior IPL deliveries (e.g., debutants)
    sub.loc[(sub["TOPSIS"] == 0) & (sub["Capped / Uncapped"] == "Capped"), "TOPSIS"] = 22.0
    sub.loc[(sub["TOPSIS"] == 0) & (sub["Capped / Uncapped"] == "Uncapped"), "TOPSIS"] = 10.0
    return sub


df_pre_18_23 = pd.concat([compute_pre_auction_topsis(y) for y in range(2018, 2024)], ignore_index=True)
acq_pre = df_pre_18_23[df_pre_18_23["Is_Sold"] == 1].copy()

# Multiple Linear Regression (with One-Hot Encoding & Train-Test Split)
X_reg = pd.get_dummies(acq_pre[["TOPSIS", "Is_OS", "Is_Capped", "Is_Retained", "Player Role"]], drop_first=True, dtype=int)
y_reg = acq_pre["Eval_Price_Lakhs"]
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
price_model = LinearRegression().fit(X_tr, y_tr)
y_pred = price_model.predict(X_te)
r2_val = r2_score(y_te, y_pred)
mae_val = mean_absolute_error(y_te, y_pred)
print(f"\nMultiple Linear Regression (Price Prediction) -> Test R2: {r2_val:.3f} | Test MAE: Rs.{mae_val:.1f} Lakhs")
for col, coef in zip(X_reg.columns, price_model.coef_):
    print(f"  {col:24s}: {coef:+8.2f} Lakhs")

# K-Means Clustering (K=3) on Open Auction Sold Players
sold_pre = acq_pre[acq_pre["Status"] == "Sold"].copy()
scaler_km = StandardScaler()
X_km = scaler_km.fit_transform(sold_pre[["Eval_Price_Lakhs", "TOPSIS"]])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_km)
sold_pre["Cluster"] = kmeans.labels_
sil = silhouette_score(X_km, kmeans.labels_)
cluster_stats = sold_pre.groupby("Cluster").agg(
    Count=("Player Name", "count"),
    Mean_Price_L=("Eval_Price_Lakhs", "mean"),
    Mean_TOPSIS=("TOPSIS", "mean"),
    OS_Pct=("Is_OS", lambda s: s.mean() * 100)
).round(1)
print(f"\nK-Means Clustering (K=3) Silhouette Score: {sil:.3f}")
print(cluster_stats)

# Phase-wise batting/bowling stats for Figure 2A
os_deliv = set(auc[auc["Is_OS"] == 1]["Deliv_Name"].dropna())
ind_deliv = set(auc[auc["Is_OS"] == 0]["Deliv_Name"].dropna())
d18_23["Bat_Origin"] = np.where(d18_23["Striker"].isin(os_deliv), "Overseas", np.where(d18_23["Striker"].isin(ind_deliv), "Indian", "Other"))
d18_23["Phase"] = np.where(d18_23["Ball"] < 6.0, "Powerplay (1-6)", np.where(d18_23["Ball"] < 15.0, "Middle (7-15)", "Death (16-20)"))
bat_ph = d18_23[d18_23["Bat_Origin"] != "Other"].groupby(["Phase", "Bat_Origin"]).agg(
    Runs=("Runs Off Bat", "sum"), Balls=("Ball", "count")
)
bat_ph["SR"] = (bat_ph["Runs"] / bat_ph["Balls"] * 100).round(2)
print("\nPhase-wise Strike Rates (2018-2023):\n", bat_ph)

# Plot Figure 2: Phase SR + K-Means Clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.patch.set_facecolor("#F8FAFC")

phases_ord = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
ind_sr = [bat_ph.loc[(p, "Indian"), "SR"] for p in phases_ord]
os_sr = [bat_ph.loc[(p, "Overseas"), "SR"] for p in phases_ord]
xp = np.arange(3)
b5 = ax1.bar(xp - w_bar/2, ind_sr, w_bar, label="Indian Batters", color="#2B6CB0", edgecolor="black")
b6 = ax1.bar(xp + w_bar/2, os_sr, w_bar, label="Overseas Batters", color="#DD6B20", edgecolor="black")
ax1.set_xticks(xp)
ax1.set_xticklabels(phases_ord, fontweight="bold")
ax1.set_ylim(0, 180)
ax1.set_ylabel("Batting Strike Rate", fontweight="bold")
ax1.set_title("2A. Phase-Wise Batting Strike Rate (2018-2023)", fontweight="bold")
ax1.legend()
ax1.grid(axis="y", linestyle=":", alpha=0.6)
for bar in list(b5) + list(b6):
    h = bar.get_height()
    ax1.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

# Label clusters dynamically by mean price & TOPSIS
order_c = cluster_stats.sort_values("Mean_Price_L").index.tolist()
label_map = {
    order_c[0]: f"Depth Picks (Avg {cluster_stats.loc[order_c[0], 'Mean_Price_L']:.0f}L | {cluster_stats.loc[order_c[0], 'OS_Pct']:.0f}% OS)",
    order_c[1]: f"High-Value Bargains (Avg {cluster_stats.loc[order_c[1], 'Mean_Price_L']:.0f}L | {cluster_stats.loc[order_c[1], 'OS_Pct']:.0f}% OS)",
    order_c[2]: f"Marquee Overpays (Avg {cluster_stats.loc[order_c[2], 'Mean_Price_L']:.0f}L | {cluster_stats.loc[order_c[2], 'OS_Pct']:.0f}% OS)"
}
c_colors = {order_c[0]: "#718096", order_c[1]: "#2E7D32", order_c[2]: "#C62828"}
for cid in order_c:
    sub_c = sold_pre[sold_pre["Cluster"] == cid]
    ax2.scatter(sub_c["Eval_Price_Lakhs"] / 100.0, sub_c["TOPSIS"],
                c=c_colors[cid], label=label_map[cid], alpha=0.72, edgecolors="black", linewidth=0.4, s=45)
ax2.set_xlabel("Auction Price (Rs. Crore)", fontweight="bold")
ax2.set_ylabel("Pre-Auction TOPSIS Performance Score", fontweight="bold")
ax2.set_title(f"2B. K-Means Value Tiers (Silhouette = {sil:.3f})", fontweight="bold")
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("outputs/fig2_phase_and_kmeans_clusters.png", dpi=200, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# STEP 4: TEAM RESULTS & MATCH WIN PREDICTION MODEL (ML CLASSIFICATION)
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print("STEP 4: TEAM PURSE STRATEGY & WIN PREDICTION MODEL (2018-2023)")
print("=" * 75)

team_map = {
    "Chennai Super Kings": "CSK", "Delhi Capitals": "DC", "Delhi Daredevils": "DC",
    "Gujarat Titans": "GT", "Kolkata Knight Riders": "KKR", "Kings XI Punjab": "PBKS",
    "Punjab Kings": "PBKS", "Lucknow Super Giants": "LSG", "Mumbai Indians": "MI",
    "Rajasthan Royals": "RR", "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB", "Sunrisers Hyderabad": "SRH"
}
m18_23 = matches[(matches["Season"] >= 2018) & (matches["Season"] <= 2023)].copy()
m18_23["T1"] = m18_23["Team 1"].map(team_map)
m18_23["T2"] = m18_23["Team 2"].map(team_map)
m18_23["Win"] = m18_23["Match Winner"].map(team_map)

tm_rows = []
for (yr, tm), grp in pd.concat([
    m18_23[["Season", "T1", "Win"]].rename(columns={"T1": "Team"}),
    m18_23[["Season", "T2", "Win"]].rename(columns={"T2": "Team"})
]).groupby(["Season", "Team"]):
    wins = (grp["Win"] == grp["Team"]).sum()
    tm_rows.append({"Season": yr, "Team": tm, "Win_Pct": wins / len(grp) * 100, "Top_Half": int(wins / len(grp) >= 0.5)})
win_df = pd.DataFrame(tm_rows)

acq_tm = acq_18_23.copy()
acq_tm["Team"] = acq_tm["Team"].replace({"DD": "DC", "KXIP": "PBKS"})
piv = acq_tm.pivot_table(index=["Season", "Team"], columns=["Player Role", "Is_OS"], values="Eval_Price_Lakhs", aggfunc="sum", fill_value=0)
piv.columns = [f"{r}_OS" if o == 1 else f"{r}_IND" for r, o in piv.columns]
piv = piv.reset_index()
tot_tm = acq_tm.groupby(["Season", "Team"]).agg(
    Total_Spend=("Eval_Price_Lakhs", "sum"),
    OS_Spend=("Eval_Price_Lakhs", lambda s: s[acq_tm.loc[s.index, "Is_OS"] == 1].sum())
).reset_index()

team_merged = win_df.merge(tot_tm, on=["Season", "Team"]).merge(piv, on=["Season", "Team"])
team_merged["OS_Spend_Pct"] = team_merged["OS_Spend"] / team_merged["Total_Spend"] * 100
for c in ["Batter_OS", "Bowler_OS", "Allrounder_OS", "WK-Batter_OS", "Allrounder_IND"]:
    team_merged[c + "_Pct"] = team_merged[c] / team_merged["Total_Spend"] * 100

comp_cols = ["OS_Spend_Pct", "Batter_OS_Pct", "Bowler_OS_Pct", "Allrounder_OS_Pct", "Allrounder_IND_Pct"]
print("\nWinning (>=50%) vs Losing (<50%) Team Purse Allocation (N=52 team-seasons):")
grp_win = team_merged.groupby("Top_Half")[comp_cols].mean().round(2)
print(grp_win)
print("Correlations with Win %:\n", team_merged[["Win_Pct"] + comp_cols].corr()["Win_Pct"].round(3))

# Match-Level Win Prediction Model
d18_23["Bat_OS"] = d18_23["Striker"].isin(os_deliv).astype(int)
d18_23["Bowl_OS"] = d18_23["Bowler"].isin(os_deliv).astype(int)
d18_23["Is_Death"] = (d18_23["Ball"] >= 15.0).astype(int)

bat_m = d18_23.groupby(["Match ID", "Batting Team", "Bat_OS", "Is_Death"])["Runs Off Bat"].sum().unstack(level=[2, 3], fill_value=0)
bat_m.columns = ["IND_O1_15", "IND_Death", "OS_O1_15", "OS_Death"]
bat_m = bat_m.reset_index().rename(columns={"Batting Team": "Team"})

bowl_m = d18_23.groupby(["Match ID", "Bowling Team", "Bowl_OS"])["Is_Wicket"].sum().unstack(fill_value=0)
bowl_m.columns = ["IND_Wkts", "OS_Wkts"]
bowl_m = bowl_m.reset_index().rename(columns={"Bowling Team": "Team"})

m_norm = m18_23[m18_23["Result"] == "normal"][["Match ID", "Match Winner"]]
match_ml = bat_m.merge(bowl_m, on=["Match ID", "Team"], how="left").fillna(0).merge(m_norm, on="Match ID")
match_ml["Won"] = (match_ml["Team"] == match_ml["Match Winner"]).astype(int)

feats = ["IND_O1_15", "OS_O1_15", "IND_Death", "OS_Death", "IND_Wkts", "OS_Wkts"]
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(match_ml[feats], match_ml["Won"], test_size=0.2, random_state=42, stratify=match_ml["Won"])
sc_m = StandardScaler()
Xm_tr_s, Xm_te_s = sc_m.fit_transform(Xm_tr), sc_m.transform(Xm_te)

clf_lr = LogisticRegression(random_state=42).fit(Xm_tr_s, ym_tr)
clf_dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xm_tr, ym_tr)
clf_knn = KNeighborsClassifier(n_neighbors=9).fit(Xm_tr_s, ym_tr)

acc_lr = accuracy_score(ym_te, clf_lr.predict(Xm_te_s))
acc_dt = accuracy_score(ym_te, clf_dt.predict(Xm_te))
acc_knn = accuracy_score(ym_te, clf_knn.predict(Xm_te_s))
print(f"\nMatch Win Prediction Test Accuracy -> Logistic Regression: {acc_lr:.3f} | KNN: {acc_knn:.3f} | Decision Tree: {acc_dt:.3f}")
for f, c in zip(feats, clf_lr.coef_[0]):
    print(f"  {f:12s}: {c:+.3f}")

# Plot Figure 3: Winning vs Losing Purse Allocation & Logistic Win Coefficients
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.patch.set_facecolor("#F8FAFC")

share_labels = ["Total OS\nSpend %", "OS Batter\nSpend %", "OS Bowler\nSpend %", "OS Allrnd\nSpend %", "IND Allrnd\nSpend %"]
lose_vals = [grp_win.loc[0, c] for c in comp_cols]
win_vals = [grp_win.loc[1, c] for c in comp_cols]
xs = np.arange(len(comp_cols))
b7 = ax1.bar(xs - w_bar/2, lose_vals, w_bar, label="Losing Teams (<50% Wins)", color="#E53E3E", alpha=0.85, edgecolor="black")
b8 = ax1.bar(xs + w_bar/2, win_vals, w_bar, label="Winning Teams (>=50% Wins)", color="#2E7D32", alpha=0.85, edgecolor="black")
ax1.set_xticks(xs)
ax1.set_xticklabels(share_labels, fontsize=9.5, fontweight="bold")
ax1.set_ylabel("Share of Total Purse (%)", fontweight="bold")
ax1.set_title("3A. Purse Strategy: Winning vs Losing Teams (2018-2023)", fontweight="bold")
ax1.legend()
ax1.grid(axis="y", linestyle=":", alpha=0.6)
for bar in list(b7) + list(b8):
    h = bar.get_height()
    ax1.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

coefs = clf_lr.coef_[0]
c_bars = ["#2B6CB0" if "IND" in f else "#DD6B20" for f in feats]
ax2.barh(feats, coefs, color=c_bars, edgecolor="black", height=0.55)
ax2.axvline(0, color="black", linewidth=1.2)
ax2.set_xlim(-0.24, 1.32)
ax2.set_xlabel("Standardized Win Impact (Log-Odds Coefficient)", fontweight="bold")
ax2.set_title(f"3B. Win Prediction Model (Logistic Reg Acc = {acc_lr*100:.1f}%)", fontweight="bold")
ax2.grid(axis="x", linestyle=":", alpha=0.6)
for idx_f, val in enumerate(coefs):
    ax2.annotate(f"{val:+.2f}", (val, idx_f), xytext=(4 if val >= 0 else -4, -3),
                 ha="left" if val >= 0 else "right",
                 textcoords="offset points", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/fig3_team_wins_and_ml_model.png", dpi=200, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# STEP 5: DEMONSTRATING 2024 PULP DECISIONS FOR SLIDE 4
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print("STEP 5: GENERATING 2024 PULP DECISION CHART (SLIDE 4)")
print("=" * 75)

from interactive_auction_model import evaluate_auction_pick

csk_2024_ret = [
    "MS Dhoni", "Ravindra Jadeja", "Ruturaj Gaikwad", "Shivam Dube",
    "Deepak Chahar", "Matheesha Pathirana", "Devon Conway", "Moeen Ali"
]
pool24 = auc[auc["Season"] == 2024].copy()
other_2024_ret = list(set(pool24[pool24["Is_Retained"] == 1]["Player Name"]) - set(csk_2024_ret))

sample_players = [
    "Shardul Thakur", "Lockie Ferguson", "Pat Cummins",
    "Nandre Burger", "Travis Head", "Shashank Singh"
]
results_s4 = []
for sp in sample_players:
    res = evaluate_auction_pick(2024, csk_2024_ret, other_2024_ret, sp)
    results_s4.append(res)
    print(f"  {sp:18s} | {res['Origin']:8s} | Decision: {res['Decision']:26s} | MaxBid: Rs.{res['Max_Bid_Lakhs']:.0f}L | ML Fair: Rs.{res['ML_Fair_Price_Lakhs']:.0f}L")

fig, ax = plt.subplots(figsize=(11.2, 4.6))
fig.patch.set_facecolor("#F8FAFC")
names_s4 = [f"{r['Player']}\n({'OS' if r['Origin']=='Overseas' else 'IND'} | {r['Role']})" for r in results_s4]
max_bids = [r["Max_Bid_Lakhs"] / 100.0 for r in results_s4]
ml_fairs = [r["ML_Fair_Price_Lakhs"] / 100.0 for r in results_s4]
dec_colors = [
    "#2E7D32" if r["Decision"] == "always overseas"
    else "#2B6CB0" if r["Decision"] == "always Indian"
    else "#DD6B20"
    for r in results_s4
]

x4 = np.arange(len(results_s4))
bars_mb = ax.bar(x4 - 0.19, max_bids, 0.36, color=dec_colors, edgecolor="black", label="PuLP Break-Even Max Bid (Cr)")
bars_ml = ax.bar(x4 + 0.19, ml_fairs, 0.36, color="#A0AEC0", edgecolor="black", label="ML Fair Price Estimate (Cr)")
ax.set_xticks(x4)
ax.set_xticklabels(names_s4, fontsize=9, fontweight="bold")
ax.set_ylabel("Price (Rs. Crore)", fontweight="bold")
ax.set_ylim(0, max(max(max_bids), max(ml_fairs)) * 1.25)
ax.set_title("4. IPL 2024 Auction Decisions & Break-Even Max Bid (PuLP + ML)", fontweight="bold")
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle=":", alpha=0.6)

for idx_r, bar in enumerate(bars_mb):
    h = bar.get_height()
    tag = (
        "always\noverseas" if results_s4[idx_r]["Decision"] == "always overseas"
        else "always\nIndian" if results_s4[idx_r]["Decision"] == "always Indian"
        else "overseas\n(constr.)"
    )
    ax.annotate(f"Rs.{h:.1f}Cr\n[{tag}]", (bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.8, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/fig4_2024_pulp_decisions.png", dpi=200, bbox_inches="tight")
plt.close()
print("\n[Done] All 4 figures generated in outputs/ successfully.")
