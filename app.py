import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from interactive_auction_model import (
    PURSE_BY_YEAR,
    get_year_constants,
    evaluate_auction_pick
)
from chase_prediction_model import predict_chase

# Page configuration
st.set_page_config(
    page_title="IPL Auction Strategy: Overseas vs Indian Optimizer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1rem;
    }
    .problem-card {
        background-color: #EDF2F7;
        border-left: 5px solid #2B6CB0;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
        color: #2D3748;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-always-os {
        background-color: #C6F6D5;
        color: #22543D;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-always-ind {
        background-color: #BEE3F8;
        color: #2A4365;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-os-constr {
        background-color: #FEEBC8;
        color: #7B341E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Problem Statement (Exact as mandated)
st.markdown('<div class="main-title">When to Pick an Overseas Player and When to Pick an Indian Player in the IPL Auction: A Data Analysis Project</div>', unsafe_allow_html=True)
st.markdown("""
<div class="problem-card">
    <b>Problem Statement:</b><br>
    In the Indian Premier League (IPL), every team can have a maximum of 8 overseas players in the squad and only 4 overseas players in the playing 11.
    Because of this rule, teams must carefully decide whether to spend money on an overseas player or an Indian player during the auction.<br>
    This project aims to study past IPL auction data, player prices, performance statistics, and team results to find clear patterns.
    The goal is to understand in which situations it is better to pick an overseas player and in which situations an Indian player is the smarter choice.
</div>
""", unsafe_allow_html=True)

# Tabs Navigation
tab_optimizer, tab_findings, tab_chase, tab_about = st.tabs([
    "🎯 Live Auction Optimizer & Max Bid",
    "📊 Empirical Findings & ML Models",
    "⚡ Match Win & Chase Predictor",
    "📘 Project Documentation & Presentation"
])

# =============================================================================
# TAB 1: LIVE AUCTION OPTIMIZER (THE CORE CAPSTONE TOOL)
# =============================================================================
with tab_optimizer:
    st.sidebar.header("Auction Setup & Parameters")

    # 1. Year Input Parameter (2018 to 2026)
    selected_year = st.sidebar.slider(
        "1. Auction Year (2018 - 2026):",
        min_value=2018,
        max_value=2026,
        value=2024,
        step=1,
        help="Once the year is selected, the system automatically loads player list, prices, and purse from the database."
    )

    # 2. Automatically load fixed constants for that year
    year_data = get_year_constants(selected_year)
    all_players = year_data["players_list"]
    total_purse_lakhs = year_data["total_purse"]
    total_purse_cr = total_purse_lakhs / 100.0
    min_spend_cr = total_purse_cr * 0.75
    pool_df = year_data["pool"]

    st.sidebar.markdown(f"""
    **Fixed System Constants for {selected_year}:**
    * **Total Team Purse:** ₹{total_purse_cr:.1f} Crore
    * **Mandatory Spend (75%):** ₹{min_spend_cr:.1f} Crore
    * **Players in Database:** {len(all_players)}
    """)

    # Default team selections
    ret_players = pool_df[pool_df["Is_Retained"] == 1]["Player Name"].tolist()
    if selected_year == 2024:
        default_same = [p for p in ["MS Dhoni", "Ravindra Jadeja", "Ruturaj Gaikwad", "Shivam Dube", "Deepak Chahar", "Matheesha Pathirana", "Devon Conway", "Moeen Ali"] if p in all_players]
    else:
        default_same = ret_players[:6] if len(ret_players) >= 6 else all_players[:4]

    default_other = [p for p in ret_players if p not in set(default_same)]

    # 4. Input Parameter: List of players sold to same team
    sold_to_same = st.sidebar.multiselect(
        "2. Players Picked by Your Team:",
        options=all_players,
        default=default_same,
        help="Players already acquired by or retained for your squad."
    )

    # 5. Input Parameter: List of players sold to other teams
    available_for_other = [p for p in all_players if p not in set(sold_to_same)]
    clean_default_other = [p for p in default_other if p in set(available_for_other)]
    sold_to_other = st.sidebar.multiselect(
        "3. Players Sold / Retained by Other Teams:",
        options=available_for_other,
        default=clean_default_other,
        help="Players off the table and unavailable for selection."
    )

    # 6. Input Parameter: Player to be sold (currently on hammer)
    available_on_hammer = [p for p in all_players if p not in set(sold_to_same) and p not in set(sold_to_other)]
    if not available_on_hammer:
        available_on_hammer = all_players

    default_hammer_name = "Pat Cummins" if "Pat Cummins" in available_on_hammer else available_on_hammer[0]

    player_on_hammer = st.sidebar.selectbox(
        "4. Player Under the Hammer:",
        options=available_on_hammer,
        index=available_on_hammer.index(default_hammer_name) if default_hammer_name in available_on_hammer else 0,
        help="The target player currently being bid on."
    )

    # Run PuLP Linear Programming Solver
    with st.spinner("Solving PuLP 0-1 Knapsack Mixed Integer Linear Program..."):
        eval_res = evaluate_auction_pick(selected_year, sold_to_same, sold_to_other, player_on_hammer)

    # Display Strategic Decision Banner
    dec = eval_res["Decision"]
    max_bid_l = eval_res["Max_Bid_Lakhs"]
    max_bid_cr = eval_res["Max_Bid_Cr"]

    col_dec, col_bid, col_val = st.columns([1.5, 1.2, 1.3])

    with col_dec:
        st.markdown("**Strategic Recommendation:**")
        if dec == "always overseas":
            st.markdown(f'<div class="badge-always-os" style="font-size:1.3rem;">✅ ALWAYS OVERSEAS</div>', unsafe_allow_html=True)
            st.caption("Selected in BOTH core unconstrained and constrained optimal squads.")
        elif dec == "always Indian":
            st.markdown(f'<div class="badge-always-ind" style="font-size:1.3rem;">🇮🇳 ALWAYS INDIAN</div>', unsafe_allow_html=True)
            st.caption("Indian options yield higher TOPSIS squad value per rupee, or overseas quota is full.")
        else:
            st.markdown(f'<div class="badge-os-constr" style="font-size:1.3rem;">⚖️ OVERSEAS DUE TO CONSTRAINTS</div>', unsafe_allow_html=True)
            st.caption("Required strictly because of squad rules, purse limits, or rival locks.")

    with col_bid:
        st.markdown("**Up to What Price Can Pick?:**")
        st.markdown(f"<h2 style='color:#1A365D; margin:0;'>₹{max_bid_cr:.2f} Cr</h2>", unsafe_allow_html=True)
        st.caption(f"Break-even limit: **₹{max_bid_l:.0f} Lakhs** (Above this, bidding hurts squad score)")

    with col_val:
        st.markdown("**Player Profile & Valuation:**")
        st.markdown(f"**{eval_res['Player']}** ({eval_res['Origin']} {eval_res['Role']})")
        st.write(f"• **Base Price:** ₹{eval_res['Base_Price_Lakhs']:.0f}L (₹{eval_res['Base_Price_Lakhs']/100:.2f} Cr)")
        st.write(f"• **ML Fair Value:** ₹{eval_res['ML_Fair_Price_Lakhs']:.0f}L | **TOPSIS:** {eval_res['TOPSIS_Score']:.1f}/100")

    st.markdown("---")

    # Squad Quota & Live Constraints Gauges
    col_g1, col_g2 = st.columns([1.2, 1])

    with col_g1:
        st.subheader("1. Live Squad & Purse Rule Compliance")
        fig_q, ax_q = plt.subplots(figsize=(8, 3.8))
        fig_q.patch.set_facecolor("#F8FAFC")
        ax_q.set_facecolor("#FFFFFF")

        labels = [
            "Squad Count (18 - 25)",
            "Indian Players (10 - 25)",
            "Overseas Players (Max 8)",
            f"Purse Spent Cr (Min 75%)"
        ]
        vals = [eval_res["Team_Count"], eval_res["Team_Indians"], eval_res["Team_Overseas"], eval_res["Purse_Spent_Cr"]]
        caps = [25, 25, 8, eval_res["Purse_Total_Cr"]]
        mins = [18, 10, 0, 0.75 * eval_res["Purse_Total_Cr"]]
        pcts = [v / c * 100 for v, c in zip(vals, caps)]

        y_pos = np.arange(len(labels))
        ax_q.barh(y_pos, [100]*4, color="#EDF2F7", edgecolor="#CBD5E0", height=0.5)
        ax_q.barh(y_pos, pcts, color="#2B6CB0", edgecolor="black", height=0.5, label="Current Locked")

        for idx_m, (m_val, c_val) in enumerate(zip(mins, caps)):
            if m_val > 0:
                ax_q.plot(m_val / c_val * 100, idx_m, marker="|", color="#E53E3E", markersize=18, markeredgewidth=2.8)
            ax_q.annotate(f"{vals[idx_m]:.1f} / {caps[idx_m]:.0f}", (min(pcts[idx_m] + 2, 75), idx_m),
                          va="center", fontsize=9.5, fontweight="bold", color="#1A202C")

        ax_q.set_yticks(y_pos)
        ax_q.set_yticklabels(labels, fontweight="bold", fontsize=9.5)
        ax_q.set_xlim(0, 105)
        ax_q.set_xlabel("% of Maximum Limit (Red mark = Mandatory Minimum Rule)", fontweight="bold")
        ax_q.grid(axis="x", linestyle=":", alpha=0.5)
        st.pyplot(fig_q)

    with col_g2:
        st.subheader("2. Price Breakdown vs Max Bid")
        fig_b, ax_b = plt.subplots(figsize=(6, 3.8))
        fig_b.patch.set_facecolor("#F8FAFC")
        ax_b.set_facecolor("#FFFFFF")

        b_labels = ["Base Price", "ML Fair Value", "Max Bid Limit"]
        b_vals = [eval_res["Base_Price_Lakhs"], eval_res["ML_Fair_Price_Lakhs"], eval_res["Max_Bid_Lakhs"]]
        b_colors = [
            "#718096", "#4299E1",
            "#2E7D32" if dec == "always overseas" else "#2B6CB0" if dec == "always Indian" else "#DD6B20"
        ]

        bars = ax_b.bar(b_labels, b_vals, color=b_colors, width=0.45, edgecolor="black")
        ax_b.set_ylabel("Price (Rs. Lakhs)", fontweight="bold")
        ax_b.set_ylim(0, max(max(b_vals) * 1.25, 120))
        ax_b.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            ax_b.annotate(f"₹{h:.0f}L\n(₹{h/100:.2f}Cr)", (bar.get_x() + bar.get_width()/2, h),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold", fontsize=9)

        st.pyplot(fig_b)

    st.info(f"**Mathematical Explanation:** {eval_res['Reason']}")

# =============================================================================
# TAB 2: EMPIRICAL FINDINGS & VISUALIZATIONS (FROM HISTORICAL IPL DATA)
# =============================================================================
with tab_findings:
    st.header("Empirical Evidence from Past IPL Auctions (2018 - 2023)")
    st.markdown("""
    All findings are derived from **1,749 auction transactions**, **295,000+ ball-by-ball deliveries**, and **match results**
    analyzed ahead of the IPL 2024 auction without intuition or guesswork.
    """)

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.subheader("1. Auction Market Price Reality & Value per Crore")
        if os.path.exists("outputs/fig1_market_prices_and_efficiency.png"):
            st.image("outputs/fig1_market_prices_and_efficiency.png", use_container_width=True)
        st.markdown("""
        * **The 3x-10x Open Auction Premium:** In open bidding, median Overseas prices are **₹2.20 Cr** (Batters/WKs) and **₹2.00 Cr** (Allrounders) compared to **₹0.20 - ₹0.60 Cr** for domestic Indian players.
        * **2x Value Efficiency:** Indian batters deliver **202.5 Runs per Crore** vs only **99.0 Runs per Crore** for Overseas batters.
        * **WK-Batter Retention Anomaly:** Overseas WK-Batters in open auctions cost less on average (**₹4.28 Cr**) than retained Indian WKs (**₹5.96 Cr**) with comparable output.
        """)

    with col_f2:
        st.subheader("2. Phase-wise Strike Rates & K-Means Value Tiers")
        if os.path.exists("outputs/fig2_phase_and_kmeans_clusters.png"):
            st.image("outputs/fig2_phase_and_kmeans_clusters.png", use_container_width=True)
        st.markdown("""
        * **Narrow Early Gap:** In Powerplay (**127.0 vs 122.3 SR**) and Middle overs (**127.9 vs 120.9 SR**), the strike rate gap is narrow (+5 to +7 points).
        * **The Death Overs Turning Point:** In overs 16-20, Overseas batters jump to **157.1 SR** (+9.5 SR gap), while Overseas fast bowlers handle **45.6%** of all death overs in the league.
        * **K-Means Clustering ($K=3$, Silhouette = 0.545):** Marquee Overpays (Avg ₹9.50 Cr) deliver the exact same TOPSIS score (**26.3**) as High-Value Bargains (Avg ₹2.19 Cr, TOPSIS **26.6**) at 4.3x the price!
        """)

    st.markdown("---")
    st.subheader("3. Team Strategy: Winning vs Losing Teams & Win Prediction Model")
    if os.path.exists("outputs/fig3_team_wins_and_ml_model.png"):
        st.image("outputs/fig3_team_wins_and_ml_model.png", use_container_width=True)

    st.markdown(r"""
    * **Winning Teams Spend Less on Overseas Overall:** Teams with $\ge 50\%$ wins spend **38.56%** of purse on Overseas players vs **42.67%** for losing teams ($r = -0.206$).
    * **Targeted Role Investment:** Winning teams invest heavily in **Overseas Bowlers** (13.9% vs 10.1%, $r = +0.182$) and **Indian Allrounders** (14.1% vs 11.6%, $r = +0.310$). Spending heavily on Overseas Batters correlates negatively with success ($r = -0.271$).
    * **Match Win Prediction:** A Logistic Regression model (74.3% accuracy) confirms Indian Overs 1-15 batting (+1.16), Bowling Wickets (+1.05 IND, +0.91 OS), and Overseas Death Runs (+0.15) drive match victories.
    """)

# =============================================================================
# TAB 3: MATCH WIN & CHASE PREDICTOR
# =============================================================================
with tab_chase:
    st.header("IPL T20 In-Match Win Prediction & Trajectory Model")
    st.markdown("Trained on **23,000+ over-by-over historical chase records** across all IPL seasons.")

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        chase_target = st.slider("Target Runs:", min_value=80, max_value=280, value=169, step=1)
        chase_overs = st.slider("Current Overs Completed:", min_value=1, max_value=18, value=6, step=1)
    with c_s2:
        chase_runs = st.slider("Current Runs Scored:", min_value=0, max_value=250, value=47, step=1)
        chase_wkts = st.slider("Current Wickets Lost:", min_value=0, max_value=9, value=3, step=1)
    with c_s3:
        future_ov = st.slider("Future Overs to Project:", min_value=max(chase_overs+1, 2), max_value=20, value=max(chase_overs+4, 10), step=1)
        chase_yr = st.selectbox("Season Match Context:", options=list(range(2024, 2010, -1)), index=0)

    chase_res = predict_chase(chase_runs, chase_wkts, chase_overs, chase_target, future_ov, chase_yr)

    if chase_res['win_probability'] >= 60.0:
        st.success(f"### {chase_res['decision']}")
    elif chase_res['win_probability'] >= 35.0:
        st.info(f"### {chase_res['decision']}")
    elif chase_res['win_probability'] >= 15.0:
        st.warning(f"### {chase_res['decision']}")
    else:
        st.error(f"### {chase_res['decision']}")

    st.markdown(f"**Need {chase_res['runs_needed']} runs in {chase_res['overs_left']} overs** | Win Probability: **{chase_res['win_probability']}%** | CRR: {chase_res['crr']} | RRR: {chase_res['rrr']}")

    fig_c, (ax_ct, ax_cg) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [2, 1]})
    ax_ct.plot([0, chase_overs], [0, chase_runs], 'b-o', lw=2.5, label=f'Current: {chase_runs}/{chase_wkts}')
    ax_ct.axhline(chase_target, color='#D32F2F', linestyle='--', lw=2, label=f'Target ({chase_target})')
    ax_ct.plot([chase_overs, 20], [chase_runs, chase_target], color='#388E3C', linestyle='--', label=f'Required (RRR {chase_res["rrr"]})')
    if future_ov > chase_overs:
        ax_ct.plot([chase_overs, future_ov], [chase_runs, chase_res['proj_future']], 'k:', lw=2, marker='s', label=f'Proj @ Over {future_ov} ({chase_res["proj_future"]})')
    ax_ct.set_xlim(0, 20)
    ax_ct.set_ylim(0, max(chase_target + 30, 220))
    ax_ct.set_xlabel('Overs', fontweight='bold')
    ax_ct.set_ylabel('Runs', fontweight='bold')
    ax_ct.legend(loc='upper left', fontsize=9)
    ax_ct.grid(True, linestyle=':', alpha=0.6)

    m_names = ['Win Prob %', 'Current RR', 'Req RR']
    m_vals = [chase_res['win_probability'], chase_res['crr'], chase_res['rrr']]
    m_cols = [chase_res['badge_color'], '#1976D2', '#D32F2F' if chase_res['rrr'] > 10 else '#F57C00']
    ax_cg.bar(m_names, m_vals, color=m_cols, width=0.45, edgecolor='black')
    ax_cg.set_ylim(0, max(110, chase_res['rrr'] * 1.3))
    ax_cg.grid(axis='y', linestyle=':', alpha=0.6)
    for bar in ax_cg.patches:
        h = bar.get_height()
        ax_cg.annotate(f'{h:.1f}', (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    st.pyplot(fig_c)

# =============================================================================
# TAB 4: METHODOLOGY & PRESENTATION DOWNLOAD
# =============================================================================
with tab_about:
    st.header("Project Methodology & Replicable Code")

    st.markdown("""
    ### Curriculum & Skill Alignment
    * **Mathematical Foundations & Linear Programming (PuLP):** Dual 0-1 Knapsack Mixed Integer Linear Program enforcing squad rules (18-25 total, 10-25 Indians, 0-8 Overseas, 75-100% purse).
    * **Accounting & Financial Management:** Opportunity cost of scarce overseas slots and break-even maximum bid price calculation via binary search.
    * **MCDM (Saaty AHP + TOPSIS):** Multi-Criteria Decision Making with Saaty pairwise comparison ($CR = 0.0 < 0.10$) and TOPSIS scoring without lookahead leakage.
    * **Data Mining & Clustering:** Preprocessing, entity resolution, and K-Means clustering ($K=3$, Silhouette = 0.545).
    * **Machine Learning:** One-Hot Encoding, 80/20 train-test split, Multiple Linear Regression (fair price), and Logistic Regression (74.3% win prediction).
    * **Python, Pandas & Matplotlib:** Clean, modular, beginner-friendly pipeline.
    """)

    st.markdown("### Download Presentation Deck")
    pptx_path = "IPL_Auction_Overseas_vs_Indian_Analysis.pptx"
    if os.path.exists(pptx_path):
        with open(pptx_path, "rb") as f:
            st.download_button(
                label="📥 Download 5-Slide Presentation (.pptx)",
                data=f,
                file_name="IPL_Auction_Overseas_vs_Indian_Analysis.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
