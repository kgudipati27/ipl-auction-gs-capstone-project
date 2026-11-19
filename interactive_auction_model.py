"""
========================================================================================
Project Title: When to Pick an Overseas Player and When to Pick an Indian Player in
               the IPL Auction: A Data Analysis Project
========================================================================================
Interactive Auction Decision & Max Bid Price Engine (2018 to 2026)

Inputs:
1) year (2018 to 2026) -> auto-loads players_list, players_with_prices, and total_purse
2) sold_to_same_team   -> list of players already picked by our team
3) sold_to_other_team  -> list of players already sold/retained by other teams
4) player_on_hammer    -> player currently under the hammer

Outputs:
1) Classification: "always overseas", "always Indian", or "overseas due to constraints"
2) Max Bid Price ("upto what price can pick?") in Rs. Lakhs & Rs. Crores
========================================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
try:
    import ipywidgets as widgets
except ImportError:
    widgets = None
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 1. FIXED SEASON PURSE TABLE (Rs. Lakhs: 1 Crore = 100 Lakhs)
# -----------------------------------------------------------------------------
PURSE_BY_YEAR = {
    2018: 8000,   # Rs. 80 Crore
    2019: 8200,   # Rs. 82 Crore
    2020: 8500,   # Rs. 85 Crore
    2021: 8500,   # Rs. 85 Crore
    2022: 9000,   # Rs. 90 Crore
    2023: 9500,   # Rs. 95 Crore
    2024: 10000,  # Rs. 100 Crore
    2025: 12000,  # Rs. 120 Crore
    2026: 12500   # Rs. 125 Crore
}

# Saaty AHP weights for [Bat_Runs, Bat_SR, Death_SR, Bowl_Wkts, Econ_Save]
A_SAATY = np.array([
    [1,   2,   2,   1,   2],
    [0.5, 1,   1,   0.5, 1],
    [0.5, 1,   1,   0.5, 1],
    [1,   2,   2,   1,   2],
    [0.5, 1,   1,   0.5, 1]
], dtype=float)
_eigvals, _eigvecs = np.linalg.eig(A_SAATY)
_w = _eigvecs[:, np.argmax(_eigvals.real)].real
AHP_WEIGHTS = _w / _w.sum()

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

_CACHE = {"auc": None, "deliv": None, "years": {}}


def _load_master_data():
    """Loads and cleans auction + deliveries CSVs once and caches them."""
    if _CACHE["auc"] is not None:
        return _CACHE["auc"], _CACHE["deliv"]

    auc = pd.read_csv("ipl_auctions_2018_2026.csv")
    deliv = pd.read_csv("IPL Deliveries Data.csv", low_memory=False)

    auc["Player Role"] = auc["Player Role"].replace({"Bowler Allrounder": "Allrounder"})
    auc["Is_Sold"] = (auc["Status"] != "Unsold").astype(int)
    auc["Is_OS"] = (auc["Player Nationality"] != "India").astype(int)
    auc["Is_Capped"] = (auc["Capped / Uncapped"] == "Capped").astype(int)
    auc["Is_Retained"] = (~auc["Status"].isin(["Sold", "Unsold"])).astype(int)
    auc = auc.sort_values(["Season", "Player Name", "Is_Sold"], ascending=[True, True, False])
    auc = auc.drop_duplicates(subset=["Season", "Player Name"], keep="first").reset_index(drop=True)
    auc["Base_Clean"] = auc["Base Price (Lakhs)"].fillna(20.0)
    auc["Eval_Price_Lakhs"] = auc["Auction / Retention Price (Lakhs)"].fillna(auc["Base_Clean"])

    deliv_players = sorted(set(deliv["Striker"].dropna()).union(set(deliv["Bowler"].dropna())))
    by_last, by_exact_lower = {}, {}
    for p in deliv_players:
        pc = p.strip()
        by_exact_lower[pc.lower()] = pc
        parts = pc.split()
        by_last.setdefault((parts[-1].lower(), parts[0][0].lower()), []).append(pc)
        by_last.setdefault((parts[-1].lower(), ""), []).append(pc)

    def _match(full_name):
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

    auc["Deliv_Name"] = auc["Player Name"].apply(_match)
    deliv["Is_Wicket"] = deliv["Wicket Type"].isin(
        ["bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"]
    ).astype(int)
    deliv["Runs_Conceded"] = deliv["Runs Off Bat"] + deliv["Extras"]

    _CACHE["auc"] = auc
    _CACHE["deliv"] = deliv
    return auc, deliv


def _compute_topsis_for_year(year, auc, deliv, lookback=5):
    """Computes pre-auction TOPSIS scores using ONLY deliveries strictly before `year`."""
    hd = deliv[(deliv["Season"] >= year - lookback) & (deliv["Season"] < year)]
    bat = hd.groupby("Striker").agg(
        Bat_Runs=("Runs Off Bat", "sum"), Bat_Balls=("Ball", "count")
    ).reset_index().rename(columns={"Striker": "Deliv_Name"})
    dbat = hd[hd["Ball"] >= 15.0].groupby("Striker").agg(
        D_Runs=("Runs Off Bat", "sum"), D_Balls=("Ball", "count")
    ).reset_index().rename(columns={"Striker": "Deliv_Name"})
    bowl = hd.groupby("Bowler").agg(
        Bowl_Balls=("Ball", "count"), Bowl_Runs=("Runs_Conceded", "sum"), Bowl_Wkts=("Is_Wicket", "sum")
    ).reset_index().rename(columns={"Bowler": "Deliv_Name"})

    sub = auc[auc["Season"] == year].copy().reset_index(drop=True)
    sub = sub.merge(bat, on="Deliv_Name", how="left").merge(dbat, on="Deliv_Name", how="left").merge(bowl, on="Deliv_Name", how="left").fillna(0)
    sub["Bat_SR"] = np.where(sub["Bat_Balls"] >= 30, sub["Bat_Runs"] / sub["Bat_Balls"] * 100, 0)
    sub["Death_SR"] = np.where(sub["D_Balls"] >= 15, sub["D_Runs"] / sub["D_Balls"] * 100, 0)
    sub["Econ_Save"] = np.where(sub["Bowl_Balls"] >= 30, np.maximum(0, 12.0 - sub["Bowl_Runs"] / (sub["Bowl_Balls"] / 6.0)), 0)

    V = sub[["Bat_Runs", "Bat_SR", "Death_SR", "Bowl_Wkts", "Econ_Save"]].values.astype(float)
    norm = np.sqrt((V ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    Vn = (V / norm) * AHP_WEIGHTS
    d_best = np.sqrt(((Vn - Vn.max(axis=0)) ** 2).sum(axis=1))
    d_worst = np.sqrt(((Vn - Vn.min(axis=0)) ** 2).sum(axis=1))
    sub["TOPSIS"] = np.round((d_worst / (d_best + d_worst + 1e-9)) * 100, 2)

    sub.loc[(sub["TOPSIS"] == 0) & (sub["Capped / Uncapped"] == "Capped"), "TOPSIS"] = 22.0
    sub.loc[(sub["TOPSIS"] == 0) & (sub["Capped / Uncapped"] == "Uncapped"), "TOPSIS"] = 10.0
    return sub


def get_year_constants(year):
    """
    Once `year` (2018-2026) is known, automatically returns:
    1) players_list
    2) players_with_prices (dict: player -> price in Lakhs)
    3) total_purse (in Lakhs)
    4) pool DataFrame (with pre-auction TOPSIS + ML Fair Price)
    """
    year = int(year)
    if year in _CACHE["years"]:
        return _CACHE["years"][year]

    # Fast path: load precomputed profile if available
    precomputed_file = "auction_player_pool_precomputed.csv"
    if os.path.exists(precomputed_file):
        df_all = pd.read_csv(precomputed_file)
        pool = df_all[df_all["Season"] == year].copy().reset_index(drop=True)
        players_list = pool["Player Name"].tolist()
        players_with_prices = dict(zip(pool["Player Name"], pool["Eval_Price_Lakhs"]))
        total_purse = PURSE_BY_YEAR[year]

        data = {
            "year": year,
            "players_list": players_list,
            "players_with_prices": players_with_prices,
            "total_purse": total_purse,
            "pool": pool
        }
        _CACHE["years"][year] = data
        return data

    auc, deliv = _load_master_data()
    pool = _compute_topsis_for_year(year, auc, deliv)

    # Train Multiple Linear Regression for Fair Price on prior years (no leakage)
    prior_years = [y for y in range(2018, year)]
    if prior_years:
        train_df = pd.concat([_compute_topsis_for_year(y, auc, deliv) for y in prior_years], ignore_index=True)
        train_df = train_df[train_df["Is_Sold"] == 1]
    else:
        # For 2018 (first year in auction CSV), fit on 2018 base/sold relationship
        train_df = pool[pool["Is_Sold"] == 1]

    feat_cols = ["TOPSIS", "Is_OS", "Is_Capped", "Is_Retained", "Player Role"]
    X_tr = pd.get_dummies(train_df[feat_cols], drop_first=True, dtype=int)
    reg = LinearRegression().fit(X_tr, train_df["Eval_Price_Lakhs"])

    X_cur = pd.get_dummies(pool[feat_cols], drop_first=True, dtype=int).reindex(columns=X_tr.columns, fill_value=0)
    pool["ML_Fair_Price_Lakhs"] = np.clip(np.round(reg.predict(X_cur), 0), pool["Base_Clean"], None)

    players_list = pool["Player Name"].tolist()
    players_with_prices = dict(zip(pool["Player Name"], pool["Eval_Price_Lakhs"]))
    total_purse = PURSE_BY_YEAR[year]

    data = {
        "year": year,
        "players_list": players_list,
        "players_with_prices": players_with_prices,
        "total_purse": total_purse,
        "pool": pool
    }
    _CACHE["years"][year] = data
    return data


# -----------------------------------------------------------------------------
# 2. PULP LINEAR PROGRAMMING SQUAD SOLVER
# -----------------------------------------------------------------------------
def _solve_squad_lp(pool, total_purse, same_team, other_team,
                    forced_player=None, forced_val=None, custom_price=None, unconstrained=False):
    """
    Solves 0-1 Integer Linear Program using PuLP (HiGHS):
    Maximize sum(TOPSIS_i * x_i)
    Subject to:
      1) 18 <= sum(x_i) <= 25             (Squad size rule)
      2) 10 <= sum(Indian x_i) <= 25      (Min Indian players)
      3)  0 <= sum(Overseas x_i) <= 8     (Max Overseas players)
      4) 0.75 * Purse <= Spend <= Purse   (Min 75% spend rule)
      5) same_team locked IN, other_team locked OUT
    """
    prob = pulp.LpProblem("IPL_Auction_Optimization", pulp.LpMaximize)
    n = len(pool)
    x = prob.add_variable_dict("p", range(n), cat="Binary")

    scores = pool["TOPSIS"].values
    prices = pool["Eval_Price_Lakhs"].values.copy()
    is_os = pool["Is_OS"].values
    names = pool["Player Name"].values

    if custom_price is not None and forced_player is not None:
        idx = np.where(names == forced_player)[0]
        if len(idx) > 0:
            prices[idx[0]] = custom_price

    prob += pulp.lpSum(scores[i] * x[i] for i in range(n))
    prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) <= total_purse

    same_set = set(same_team)
    other_set = set(other_team) - same_set

    for i in range(n):
        if names[i] in same_set:
            prob += x[i] == 1
        elif names[i] in other_set:
            prob += x[i] == 0

    if not unconstrained:
        prob += pulp.lpSum(x[i] for i in range(n)) >= 18
        prob += pulp.lpSum(x[i] for i in range(n)) <= 25
        prob += pulp.lpSum(x[i] for i in range(n) if is_os[i] == 0) >= 10
        prob += pulp.lpSum(x[i] for i in range(n) if is_os[i] == 1) <= 8
        prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) >= 0.75 * total_purse
    else:
        # Unconstrained core picks: relax 8-overseas cap, 10-Indian min, and 75% spend floor
        prob += pulp.lpSum(x[i] for i in range(n)) <= min(25, max(11, len(same_set) + 4))

    if forced_player is not None and forced_val is not None:
        idx = np.where(names == forced_player)[0]
        if len(idx) > 0:
            prob += x[idx[0]] == forced_val

    try:
        res = prob.solve(pulp.HiGHS(msg=False))
    except Exception:
        try:
            res = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            res = prob.solve()
    if res.status != pulp.LpSolveStatus.Optimal:
        return False, -1.0, []
    chosen = [names[i] for i in range(n) if pulp.value(x[i]) > 0.5]
    return True, float(res.objective), chosen


def evaluate_auction_pick(year, sold_to_same_team, sold_to_other_team, player_on_hammer):
    """
    Evaluates `player_on_hammer` in `year` given `sold_to_same_team` and `sold_to_other_team`.
    Returns classification ("always overseas", "always Indian", "overseas due to constraints")
    and maximum bid price ("upto what price can pick?").
    """
    data = get_year_constants(year)
    pool = data["pool"]
    total_purse = data["total_purse"]

    # Ensure hammer player is not accidentally blocked by same/other lists during evaluation
    same_list = [p for p in sold_to_same_team if p != player_on_hammer]
    other_list = [p for p in sold_to_other_team if p != player_on_hammer]

    row = pool[pool["Player Name"] == player_on_hammer].iloc[0]
    is_os = int(row["Is_OS"])
    origin = "Overseas" if is_os == 1 else "Indian"
    role = row["Player Role"]
    base_p = float(row["Base_Clean"])
    topsis_v = float(row["TOPSIS"])
    ml_fair_p = float(row["ML_Fair_Price_Lakhs"])

    same_df = pool[pool["Player Name"].isin(same_list)]
    n_curr = len(same_df)
    os_curr = int(same_df["Is_OS"].sum())
    ind_curr = n_curr - os_curr
    spent_curr = float(same_df["Eval_Price_Lakhs"].sum())
    purse_rem = total_purse - spent_curr

    # Solve Constrained & Unconstrained PuLP models at player's Base Price
    ok_w, z_with, ch_with = _solve_squad_lp(pool, total_purse, same_list, other_list, player_on_hammer, 1, base_p, False)
    ok_wo, z_without, ch_wo = _solve_squad_lp(pool, total_purse, same_list, other_list, player_on_hammer, 0, base_p, False)
    u_ok_w, u_z_w, _ = _solve_squad_lp(pool, total_purse, same_list, other_list, player_on_hammer, 1, base_p, True)
    u_ok_wo, u_z_wo, _ = _solve_squad_lp(pool, total_purse, same_list, other_list, player_on_hammer, 0, base_p, True)

    picked_c = ok_w and (not ok_wo or z_with >= z_without - 1e-4)
    picked_u = u_ok_w and (not u_ok_wo or u_z_w >= u_z_wo - 1e-4)

    # Check if an Overseas replacement in the same role took the slot when an Indian is skipped
    os_in_role_wo = pool[
        (pool["Player Name"].isin(ch_wo)) &
        (~pool["Player Name"].isin(same_list)) &
        (pool["Player Role"] == role) &
        (pool["Is_OS"] == 1)
    ]

    if is_os == 1:
        if not ok_w or not picked_c:
            decision = "always Indian"
            reason = (
                f"Overseas quota full ({os_curr}/8) or constraints forbid."
                if not ok_w else
                "Indian options give a higher total TOPSIS squad score under the 8-overseas cap."
            )
        elif picked_u:
            decision = "always overseas"
            reason = "Selected in BOTH Unconstrained and Constrained PuLP models (elite talent edge)."
        else:
            decision = "overseas due to constraints"
            reason = "Not needed unconstrained, but required under squad/purse/rival-lock constraints."
    else:
        if ok_w and picked_c:
            decision = "always Indian"
            reason = "Indian pick improves squad TOPSIS score while saving scarce overseas slots."
        elif len(os_in_role_wo) > 0 and os_curr < 8:
            decision = "always overseas" if not picked_u else "overseas due to constraints"
            reason = f"PuLP prefers an available Overseas {role} ({os_in_role_wo.iloc[0]['Player Name']}) for higher squad impact."
        else:
            decision = "always Indian"
            reason = "Pick an Indian in this role, but skip this player above base price (better Indian value exists)."

    # Calculate "upto what price can pick?" via PuLP Break-Even Binary Search
    max_bid_lakhs = 0.0
    if ok_w and picked_c:
        lo, hi = base_p, max(base_p, purse_rem)
        for _ in range(7):
            mid = (lo + hi) / 2.0
            ok_m, z_m, _ = _solve_squad_lp(pool, total_purse, same_list, other_list, player_on_hammer, 1, mid, False)
            if ok_m and (not ok_wo or z_m >= z_without - 1e-4):
                lo = mid
            else:
                hi = mid
        max_bid_lakhs = round(lo, 0)

    return {
        "Year": year,
        "Player": player_on_hammer,
        "Origin": origin,
        "Role": role,
        "Capped": row["Capped / Uncapped"],
        "Base_Price_Lakhs": base_p,
        "Actual_Price_Lakhs": float(row["Eval_Price_Lakhs"]),
        "TOPSIS_Score": topsis_v,
        "ML_Fair_Price_Lakhs": ml_fair_p,
        "Decision": decision,
        "Max_Bid_Lakhs": max_bid_lakhs,
        "Max_Bid_Cr": round(max_bid_lakhs / 100.0, 2),
        "Reason": reason,
        "Team_Count": n_curr,
        "Team_Indians": ind_curr,
        "Team_Overseas": os_curr,
        "Purse_Total_Cr": total_purse / 100.0,
        "Purse_Spent_Cr": round(spent_curr / 100.0, 2),
        "Purse_Remaining_Cr": round(purse_rem / 100.0, 2)
    }


# -----------------------------------------------------------------------------
# 3. INTERACTIVE PLOT & WIDGETS (Matching chase_prediction_model Template)
# -----------------------------------------------------------------------------
def plot_auction_model(year, sold_to_same_team, sold_to_other_team, player_on_hammer):
    """Plots the instant PuLP decision, constraint status, and max bid price."""
    res = evaluate_auction_pick(year, list(sold_to_same_team), list(sold_to_other_team), player_on_hammer)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.patch.set_facecolor("#F8FAFC")

    color_map = {
        "always overseas": "#2E7D32",
        "always Indian": "#2B6CB0",
        "overseas due to constraints": "#DD6B20"
    }
    badge_c = color_map[res["Decision"]]

    fig.suptitle(
        f"DECISION: \"{res['Decision'].upper()}\"   |   UP TO WHAT PRICE CAN PICK?: Rs.{res['Max_Bid_Lakhs']:.0f} Lakhs (Rs.{res['Max_Bid_Cr']:.2f} Cr)\n"
        f"IPL {res['Year']} | On Hammer: {res['Player']} ({res['Origin']} {res['Role']}, {res['Capped']}) | Pre-Auction TOPSIS: {res['TOPSIS_Score']}/100",
        fontsize=11.5, fontweight="bold", color=badge_c, y=1.04
    )

    # Left: Squad Constraints Progress
    labels = ["Squad Size\n(Min 18, Max 25)", "Indians\n(Min 10, Max 25)", "Overseas\n(Max 8)", "Purse Spent Cr\n(Min 75%)"]
    vals = [res["Team_Count"], res["Team_Indians"], res["Team_Overseas"], res["Purse_Spent_Cr"]]
    caps = [25, 25, 8, res["Purse_Total_Cr"]]
    mins = [18, 10, 0, 0.75 * res["Purse_Total_Cr"]]
    pcts = [v / c * 100 for v, c in zip(vals, caps)]

    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, [100]*4, color="#EDF2F7", edgecolor="#CBD5E0", height=0.5)
    ax1.barh(y_pos, pcts, color="#4A5568", edgecolor="black", height=0.5, label="Current Locked")
    for idx_m, (m_val, c_val) in enumerate(zip(mins, caps)):
        if m_val > 0:
            ax1.plot(m_val / c_val * 100, idx_m, marker="|", color="#E53E3E", markersize=18, markeredgewidth=2.5)
        ax1.annotate(f"{vals[idx_m]:.1f} / {caps[idx_m]:.0f}", (min(pcts[idx_m] + 2, 78), idx_m),
                     va="center", fontsize=9, fontweight="bold", color="#1A202C")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontweight="bold", fontsize=9)
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("% of Maximum Limit (Red mark = Minimum Rule)", fontweight="bold")
    ax1.set_title("1. Live IPL Squad & Purse Constraints", fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # Right: Price Comparison (Base Price vs ML Fair Price vs PuLP Break-Even Max Bid)
    p_cats = ["Base Price", "ML Fair Value", "PuLP Max Bid\n(Can Pick Up To)"]
    p_vals = [res["Base_Price_Lakhs"], res["ML_Fair_Price_Lakhs"], res["Max_Bid_Lakhs"]]
    p_cols = ["#718096", "#4299E1", badge_c]
    bars = ax2.bar(p_cats, p_vals, color=p_cols, width=0.45, edgecolor="black")
    ax2.set_ylabel("Price (Rs. Lakhs)", fontweight="bold")
    ax2.set_title("2. Valuation & Break-Even Bid Limit", fontweight="bold")
    ax2.set_ylim(0, max(max(p_vals) * 1.25, 100))
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"Rs.{h:.0f}L\n({h/100:.2f} Cr)", (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/interactive_auction_output.png", dpi=180, bbox_inches="tight")
    try:
        get_ipython()
        plt.show()
    except NameError:
        pass
    plt.close()

    print(f"Decision: {res['Decision']} | Max Bid: Rs.{res['Max_Bid_Lakhs']:.0f} Lakhs (Rs.{res['Max_Bid_Cr']:.2f} Cr)")
    print(f"Purse: Rs.{res['Purse_Spent_Cr']} Cr spent / Rs.{res['Purse_Total_Cr']} Cr total (Remaining: Rs.{res['Purse_Remaining_Cr']} Cr)")
    print(f"Why: {res['Reason']}")


def interactive_auction_model(year, sold_to_same_team, sold_to_other_team, player_on_hammer):
    plot_auction_model(year, sold_to_same_team, sold_to_other_team, player_on_hammer)


# Initialize widgets for 2024 default
_init_data = get_year_constants(2024)
_init_pool = _init_data["pool"]
_init_same = ["MS Dhoni", "Ravindra Jadeja", "Ruturaj Gaikwad", "Shivam Dube", "Deepak Chahar", "Matheesha Pathirana", "Devon Conway", "Moeen Ali"]
_init_other = sorted(list(set(_init_pool[_init_pool["Is_Retained"] == 1]["Player Name"]) - set(_init_same)))
_init_avail = [p for p in _init_data["players_list"] if p not in set(_init_same) and p not in set(_init_other)]
_default_hammer = "Pat Cummins" if "Pat Cummins" in _init_avail else _init_avail[0]

if widgets is not None:
    year_slider = widgets.IntSlider(
        value=2024, min=2018, max=2026, step=1,
        description="Auction Year:", continuous_update=False,
        style={"description_width": "initial"}
    )
    sold_to_same_widget = widgets.SelectMultiple(
        options=_init_data["players_list"],
        value=tuple(p for p in _init_same if p in _init_data["players_list"]),
        description="Sold to Same Team:",
        rows=6, style={"description_width": "initial"}
    )
    sold_to_other_widget = widgets.SelectMultiple(
        options=_init_data["players_list"],
        value=tuple(p for p in _init_other if p in _init_data["players_list"]),
        description="Sold to Other Team:",
        rows=6, style={"description_width": "initial"}
    )
    player_on_hammer_dropdown = widgets.Dropdown(
        options=_init_data["players_list"],
        value=_default_hammer,
        description="Player on Hammer:",
        style={"description_width": "initial"}
    )


    def _on_year_change(change):
        yr = int(change["new"])
        d = get_year_constants(yr)
        plist = d["players_list"]
        p_df = d["pool"]
        ret = p_df[p_df["Is_Retained"] == 1]["Player Name"].tolist()
        same_def = tuple(ret[:6]) if len(ret) >= 6 else tuple(plist[:4])
        other_def = tuple(p for p in ret if p not in set(same_def))
        sold_to_same_widget.options = plist
        sold_to_same_widget.value = same_def
        sold_to_other_widget.options = plist
        sold_to_other_widget.value = other_def
        player_on_hammer_dropdown.options = plist
        rem = [p for p in plist if p not in set(same_def) and p not in set(other_def)]
        player_on_hammer_dropdown.value = rem[0] if rem else plist[0]


    year_slider.observe(_on_year_change, names="value")

    interactive_output = widgets.interactive_output(
        interactive_auction_model,
        {
            "year": year_slider,
            "sold_to_same_team": sold_to_same_widget,
            "sold_to_other_team": sold_to_other_widget,
            "player_on_hammer": player_on_hammer_dropdown
        }
    )

    controls = widgets.VBox([
        year_slider,
        widgets.HBox([sold_to_same_widget, sold_to_other_widget]),
        player_on_hammer_dropdown
    ])

if __name__ == "__main__":
    print("[Demo] Running interactive_auction_model for IPL 2024 (Pat Cummins on hammer)...")
    plot_auction_model(2024, _init_same, _init_other, "Pat Cummins")
