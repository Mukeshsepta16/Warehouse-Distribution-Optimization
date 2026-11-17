"""Warehouse Distribution Optimization - Streamlit app (Operations Research mini project)."""
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from transport import run_all, total_cost, balance, solve_linprog

st.set_page_config(page_title="Warehouse Distribution Optimization", page_icon="🚚", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1500px}
[data-testid="stSidebar"]{background:#082030;border-right:1px solid #1d3a4a}
.logo{width:56px;height:56px;border-radius:12px;background:#fff;display:flex;align-items:center;justify-content:center;font-size:30px}
.title{font-size:2rem;font-weight:500;line-height:56px;color:#f2f7fa}
.title span{color:#8fb3c4;font-size:1.2rem}
.badge{margin-top:8px;text-align:center;border:1px solid #1e90ff;border-radius:8px;padding:10px;color:#cfe6f5;background:#0c2c40}
.kcard{display:flex;align-items:center;background:linear-gradient(160deg,#0f3446,#0b2735);border:1px solid #2c4a5a;border-radius:14px;height:140px;padding:6px 4px}
.kcard.green{background:linear-gradient(160deg,#18422f,#103323);border-color:#2d6a49}
.kv{flex:1;text-align:center;border-right:1px solid #2c4a5a;padding:0 6px}
.kv:last-child{border-right:none}
.kv .n{font-size:1.9rem;font-weight:700;color:#fff;white-space:nowrap}
.kv .n small{font-size:.9rem;font-weight:400;color:#9fb8c6}
.kv .l{font-size:.82rem;color:#b9cbd6;margin-top:2px}
.kcard.green .l{color:#41d96a}
div[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(160deg,#0f3446,#0a2431);border:1px solid #2c4a5a!important;border-radius:14px}
.big{font-size:3rem;font-weight:600;color:#fff;line-height:1.1}
.sub{color:#b9cbd6;font-size:1.05rem;margin-bottom:14px}
.dot{display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:10px;vertical-align:middle}
.leg{color:#e8f1f5;font-size:1.05rem;margin:10px 0}
</style>
""", unsafe_allow_html=True)

def kpi(items, green=False):
    cells = "".join(f'<div class="kv"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in items)
    return f'<div class="kcard{" green" if green else ""}">{cells}</div>'

def arc(lon0, lat0, lon1, lat1, n=40, bulge=0.18):
    t = np.linspace(0, 1, n)
    mx, my = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    dx, dy = lon1 - lon0, lat1 - lat0
    cx, cy = mx - dy * bulge, my + dx * bulge
    return ((1 - t) ** 2 * lon0 + 2 * (1 - t) * t * cx + t ** 2 * lon1,
            (1 - t) ** 2 * lat0 + 2 * (1 - t) * t * cy + t ** 2 * lat1)


# ---------- data ----------
@st.cache_data
def load():
    return (pd.read_csv("data/warehouses.csv"), pd.read_csv("data/stores.csv"),
            pd.read_csv("data/distances.csv"), pd.read_csv("data/baseline_plan.csv"))
W0, S0, DIST, BASE = load()

st.sidebar.title("🚚 Warehouse Distribution")
page = st.sidebar.radio("Go to", ["0. Dashboard", "1. Overview", "2. Data", "3. Transportation Methods",
                                  "4. Optimization & Before/After", "5. Scenario Analysis", "6. Results & Download"])
st.sidebar.divider()
rate = st.sidebar.number_input("Rate (₹ per unit per km)", 0.01, 1.0, 0.04, 0.01,
                               help="Default 0.04 ≈ DPIIT-NCAER (2025) average road cost of Rs 3.78 per tonne-km × an assumed 10 kg carton (0.01 tonne). Changing it scales all costs but not the optimal plan or the % saving.")
st.sidebar.caption("Illustrative data. Not real company records.")

wn, sn = list(W0.warehouse), list(S0.store)
dist_m = DIST.pivot(index="warehouse", columns="store", values="distance_km").loc[wn, sn]
cost_m = (dist_m * rate).round(2)
supply, demand = list(W0.capacity_units), list(S0.demand_units)

def plan_df(x, rows, cols):
    return pd.DataFrame(x[:len(rows), :len(cols)], index=rows, columns=cols)

# baseline (assumed) plan
base_x = np.zeros((len(wn), len(sn)))
for _, r in BASE.iterrows(): base_x[wn.index(r.warehouse), sn.index(r.store)] = r.units
base_cost = total_cost(cost_m.values, base_x)

res = run_all(cost_m.values, supply, demand)
opt_x = res["LP"][:len(wn), :len(sn)]
opt_cost = res["lp_cost"]
savings = base_cost - opt_cost
pct = savings / base_cost * 100 if base_cost else 0

# ---------- pages ----------
if page.startswith("0"):
    h1, h2, h3 = st.columns([0.7, 7, 3])
    h1.markdown('<div class="logo">🚚</div>', unsafe_allow_html=True)
    h2.markdown('<div class="title">Warehouse Distribution Analytics Hub <span>– Illustrative Case</span></div>', unsafe_allow_html=True)
    h3.markdown(f'<div class="badge">Illustrative data · ₹{rate} per unit-km</div>', unsafe_allow_html=True)

    util = float((opt_x.sum(1) / np.array(supply) * 100).mean())
    fulfil = float((opt_x.sum(0) / np.array(demand) * 100).mean())
    routes = int((opt_x > 1e-9).sum())
    k = st.columns([2.2, 2, 3, 1.6])
    k[0].markdown(kpi([(f"₹{base_cost:,.0f}", "Original Cost (assumed)"), (f"₹{opt_cost:,.0f}", "Optimized Cost")]), unsafe_allow_html=True)
    k[1].markdown(kpi([(f"{sum(supply):,}<small> units</small>", "Total Supply"), (f"{sum(demand):,}<small> units</small>", "Total Demand")]), unsafe_allow_html=True)
    k[2].markdown(kpi([(f"{util:.0f}%", "Warehouse Utilization"), (f"{fulfil:.0f}%", "Demand Fulfilled"), (f"{routes}", "Routes Used")]), unsafe_allow_html=True)
    k[3].markdown(kpi([(f"₹{savings:,.0f}", f"Saved ({pct:.1f}%)")], green=True), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1.15])
    palette = ["#f77f1f", "#1e7bff", "#e6e6e6", "#41d96a", "#c05cff"]
    with left:
        with st.container(border=True):
            t1, t2 = st.columns([3, 2])
            t1.markdown("#### Optimized Cost by Warehouse")
            view = t2.selectbox("View", ["Cost (₹)", "Units"], label_visibility="collapsed")
            wcost = (opt_x * cost_m.values).sum(1); wunits = opt_x.sum(1)
            vals = wcost if view.startswith("Cost") else wunits
            a, b = st.columns([1, 1.5])
            total_txt = f"₹{wcost.sum():,.0f}" if view.startswith("Cost") else f"{wunits.sum():,.0f}"
            a.markdown(f'<div class="big">{total_txt}</div><div class="sub">{routes} routes · {wunits.sum():,.0f} units</div>', unsafe_allow_html=True)
            a.markdown("".join(f'<div class="leg"><span class="dot" style="background:{palette[i]}"></span>{w}</div>' for i, w in enumerate(wn)), unsafe_allow_html=True)
            pie = go.Figure(go.Pie(labels=wn, values=vals, hole=0.55, sort=False, marker=dict(colors=palette[:len(wn)]),
                                   textinfo="label+percent", textposition="outside", textfont=dict(color="#e8f1f5")))
            pie.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            b.plotly_chart(pie, theme=None, width="stretch")
    with right:
        with st.container(border=True):
            st.markdown("#### Distribution Network – Optimized Routes")
            fig = go.Figure()
            top = sorted(((opt_x[i, j], i, j) for i in range(len(wn)) for j in range(len(sn)) if opt_x[i, j] > 1e-9), reverse=True)
            for q, i, j in top:
                lo, la = arc(W0.lon[i], W0.lat[i], S0.lon[j], S0.lat[j])
                fig.add_trace(go.Scattergeo(lon=lo, lat=la, mode="lines", line=dict(width=1 + q / 60, color="rgba(255,255,255,0.85)"),
                              hoverinfo="text", text=f"{wn[i]} → {sn[j]}: {q:.0f} units · ₹{q * cost_m.iloc[i, j]:,.0f}", showlegend=False))
            for q, i, j in top[:3]:
                lo, la = arc(W0.lon[i], W0.lat[i], S0.lon[j], S0.lat[j], n=3)
                fig.add_trace(go.Scattergeo(lon=[lo[1]], lat=[la[1]], mode="text", showlegend=False, hoverinfo="skip",
                              text=f"<b>₹{q * cost_m.iloc[i, j]:,.0f}</b><br>{q:.0f} units", textfont=dict(color="#4dd0ff", size=13)))
            fig.add_trace(go.Scattergeo(lon=W0.lon, lat=W0.lat, text=[w.upper()[:3] for w in wn], mode="markers+text", textposition="top center",
                          textfont=dict(color="#fff", size=12), marker=dict(size=12, color="#1e90ff", line=dict(width=2, color="#fff")), name="Warehouse", hovertext=wn, hoverinfo="text"))
            fig.add_trace(go.Scattergeo(lon=S0.lon, lat=S0.lat, text=[x.upper()[:3] for x in sn], mode="markers+text", textposition="bottom center",
                          textfont=dict(color="#fff", size=12), marker=dict(size=10, color="#f77f1f", line=dict(width=2, color="#fff")), name="Store", hovertext=sn, hoverinfo="text"))
            fig.update_geos(scope="asia", resolution=50, lataxis_range=[9, 20], lonaxis_range=[72, 84], showland=True, landcolor="#1f6f3a",
                            showocean=True, oceancolor="#0c2a3a", showcountries=True, countrycolor="#0a2230", showcoastlines=False, bgcolor="rgba(0,0,0,0)")
            fig.update_layout(height=370, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(color="#e8f1f5"), orientation="h", y=0.02))
            st.plotly_chart(fig, theme=None, width="stretch")

    with st.container(border=True):
        st.markdown("#### Cost per Unit Delivered – Target vs. Optimized vs. Baseline")
        tgt = cost_m.min(axis=0).values
        act = (opt_x * cost_m.values).sum(0) / np.where(opt_x.sum(0) > 0, opt_x.sum(0), 1)
        bas = (base_x * cost_m.values).sum(0) / np.where(base_x.sum(0) > 0, base_x.sum(0), 1)
        ln = go.Figure()
        for nm, y, col in [("Target (cheapest route)", tgt, "#41d96a"), ("Optimized", act, "#1e90ff"), ("Baseline (assumed)", bas, "#d633ff")]:
            ln.add_trace(go.Scatter(x=sn, y=y, mode="lines+markers", name=nm, line=dict(width=3, color=col), marker=dict(size=8)))
        ln.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbe7ee"),
                         yaxis=dict(title="₹ per unit", gridcolor="#2c4a5a", rangemode="tozero"),
                         xaxis=dict(showgrid=True, gridcolor="#2c4a5a", griddash="dash"),
                         legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(ln, theme=None, width="stretch")
        st.caption("Lower is better. Target = cheapest possible per-unit cost to each store; the gap shows the cost of limited warehouse capacity and demand sharing.")

elif page.startswith("1"):
    st.title("Warehouse Distribution Optimization")
    st.caption("Operations Research mini project · Transportation Model using Linear Programming")
    c = st.columns(4)
    c[0].metric("Warehouses", len(wn)); c[1].metric("Destinations", len(sn))
    c[2].metric("Total supply", f"{sum(supply):,}"); c[3].metric("Total demand", f"{sum(demand):,}")
    c = st.columns(4)
    c[0].metric("Original cost (assumed plan)", f"₹{base_cost:,.0f}")
    c[1].metric("Optimized cost", f"₹{opt_cost:,.0f}")
    c[2].metric("Savings", f"₹{savings:,.0f}", f"{pct:.2f}%")
    c[3].metric("Status", "Optimal ✅" if "Optimization terminated" in res["lp_msg"] else res["lp_msg"][:20])
    st.subheader("The problem")
    st.write("A company ships FMCG goods from **3 warehouses** (Bengaluru, Chennai, Hyderabad) to **4 retail "
             "distribution centres** (Mysuru, Mangaluru, Coimbatore, Vijayawada). Each warehouse has limited stock and "
             "each centre needs a fixed quantity. **Question: how many units should go on each route so the total "
             "transportation cost is minimum?**")
    st.subheader("The model")
    st.latex(r"\text{Min } Z=\sum_i\sum_j C_{ij}X_{ij}")
    st.latex(r"\sum_j X_{ij}\le S_i \quad(\text{supply}),\qquad \sum_i X_{ij}=D_j \quad(\text{demand}),\qquad X_{ij}\ge 0")
    st.info("**Data rule:** warehouse capacities, demands, distances and the baseline plan are illustrative "
            "(research-based assumptions), not observed company records. For background statistics on India's logistics "
            "cost, cite the official 2025 DPIIT–NCAER assessment directly from its source.")

elif page.startswith("2"):
    st.title("Data")
    a, b = st.columns(2)
    a.subheader("Warehouse capacity"); a.dataframe(W0[["warehouse", "capacity_units"]], hide_index=True, width="stretch")
    b.subheader("Store demand"); b.dataframe(S0[["store", "demand_units"]], hide_index=True, width="stretch")
    st.subheader("Distance (km) – approximate road distances, verify on Google Maps")
    st.dataframe(dist_m, width="stretch")
    st.subheader(f"Cost per unit (₹) = distance × ₹{rate}")
    st.dataframe(cost_m, width="stretch")
    st.subheader("Cost matrix heatmap")
    st.plotly_chart(px.imshow(cost_m, text_auto=True, color_continuous_scale="Blues", aspect="auto"), width="stretch")
    st.caption(res["note"])

elif page.startswith("3"):
    st.title("Transportation Methods")
    st.write("Each method finds a starting plan. MODI then tests it and improves it until no route can lower the cost.")
    names = {"NWC": "North-West Corner", "LCM": "Least Cost", "VAM": "Vogel's Approximation (VAM)",
             "MODI_from_NWC": "MODI (improved from NWC)", "LP": "SciPy linprog"}
    rows = [{"Method": names[k], "Total cost (₹)": total_cost(res["cost_bal"], res[k])} for k in names]
    cmp = pd.DataFrame(rows)
    st.dataframe(cmp, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(cmp, x="Method", y="Total cost (₹)", text_auto=",.0f", color="Method"), width="stretch")
    for k in names:
        with st.expander(f"{names[k]} – allocation"):
            st.dataframe(plan_df(res[k], wn + (["Dummy"] if len(res[k]) > len(wn) else []),
                                 sn + (["Dummy"] if res[k].shape[1] > len(sn) else [])), width="stretch")
    st.subheader("MODI iterations (u–v method)")
    rn = wn + (["Dummy"] if len(res["cost_bal"]) > len(wn) else [])
    cn = sn + (["Dummy"] if res["cost_bal"].shape[1] > len(sn) else [])
    for t, lg in enumerate(res["modi_log"], 1):
        st.markdown(f"**Iteration {t}** · cost ₹{lg['cost']:,.0f} · u = {lg['u']} · v = {lg['v']}")
        st.dataframe(pd.DataFrame(lg["d"], index=rn, columns=cn).round(2), width="stretch")
        ei, ej = lg["entering"]
        if lg["dmin"] < -1e-9:
            st.warning(f"Most negative dij = {lg['dmin']:.2f} at {rn[ei]} → {cn[ej]}. Not optimal: shift units around the loop.")
        else:
            st.success("All dij ≥ 0 → this solution is optimal.")

elif page.startswith("4"):
    st.title("Optimization & Before/After Comparison")
    st.caption("Baseline = ASSUMED fixed-route habit plan (not observed company data).")
    c = st.columns(4)
    c[0].metric("Original cost", f"₹{base_cost:,.0f}"); c[1].metric("Optimized cost", f"₹{opt_cost:,.0f}")
    c[2].metric("Savings", f"₹{savings:,.0f}"); c[3].metric("Reduction", f"{pct:.2f}%")
    st.latex(r"\text{Savings}=\text{Original}-\text{Optimized},\quad \text{Reduction \%}=\frac{\text{Original}-\text{Optimized}}{\text{Original}}\times100")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(x=["Original", "Optimized"], y=[base_cost, opt_cost], text=[f"₹{base_cost:,.0f}", f"₹{opt_cost:,.0f}"],
                          labels={"x": "", "y": "Total cost (₹)"}, title="Cost comparison", color=["Original", "Optimized"]), width="stretch")
    b.plotly_chart(px.imshow(plan_df(opt_x, wn, sn), text_auto=True, color_continuous_scale="Greens",
                             title="Optimal allocation (units)", aspect="auto"), width="stretch")
    util = pd.DataFrame({"Warehouse": wn * 2, "Plan": ["Before"] * len(wn) + ["After"] * len(wn),
                         "Utilization %": list(base_x.sum(1) / np.array(supply) * 100) + list(opt_x.sum(1) / np.array(supply) * 100)})
    fill = pd.DataFrame({"Store": sn * 2, "Plan": ["Before"] * len(sn) + ["After"] * len(sn),
                         "Demand met %": list(base_x.sum(0) / np.array(demand) * 100) + list(opt_x.sum(0) / np.array(demand) * 100)})
    a, b = st.columns(2)
    a.plotly_chart(px.bar(util, x="Warehouse", y="Utilization %", color="Plan", barmode="group", title="Warehouse utilization"), width="stretch")
    b.plotly_chart(px.bar(fill, x="Store", y="Demand met %", color="Plan", barmode="group", title="Demand fulfillment"), width="stretch")
    st.subheader("Distribution network map (optimal plan)")
    fig = go.Figure()
    for i, w in enumerate(wn):
        for j, s in enumerate(sn):
            if opt_x[i, j] > 0:
                fig.add_trace(go.Scattergeo(lon=[W0.lon[i], S0.lon[j]], lat=[W0.lat[i], S0.lat[j]], mode="lines",
                              line=dict(width=1 + opt_x[i, j] / 40, color="#2a9d8f"), hoverinfo="text",
                              text=f"{w} → {s}: {opt_x[i, j]:.0f} units", showlegend=False))
    fig.add_trace(go.Scattergeo(lon=W0.lon, lat=W0.lat, text=wn, mode="markers+text", name="Warehouse", textposition="top center", marker=dict(size=13, color="#e76f51", symbol="square")))
    fig.add_trace(go.Scattergeo(lon=S0.lon, lat=S0.lat, text=sn, mode="markers+text", name="Store", textposition="bottom center", marker=dict(size=11, color="#264653")))
    fig.update_geos(scope="asia", lataxis_range=[9, 20], lonaxis_range=[72, 84], showcountries=True, showland=True, landcolor="#f1f1f1")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.subheader("Baseline (assumed) plan")
    st.dataframe(plan_df(base_x, wn, sn), width="stretch")

elif page.startswith("5"):
    st.title("Scenario Analysis")
    st.write("Edit capacity, demand or costs, then see the re-optimized plan. Shortfalls are handled with a dummy warehouse (unmet demand); surplus with a dummy destination.")
    out = st.selectbox("Warehouse to make unavailable in Scenario 5", wn, index=1)
    g = st.columns(3)
    dem_up = g[0].slider("Demand increase % (Scenario 2)", 0, 100, 20)
    cap_dn = g[1].slider("Capacity reduction % (Scenario 3)", 0, 80, 20)
    cost_up = g[2].slider("Transport cost increase % (Scenario 4)", 0, 100, 25)
    scen = {
        "1. Normal demand": (supply, demand, 1.0),
        f"2. Demand +{dem_up}%": (supply, [d * (1 + dem_up / 100) for d in demand], 1.0),
        f"3. Capacity -{cap_dn}%": ([s * (1 - cap_dn / 100) for s in supply], demand, 1.0),
        f"4. Cost +{cost_up}%": (supply, demand, 1 + cost_up / 100),
        f"5. {out} unavailable": ([0 if w == out else s for w, s in zip(wn, supply)], demand, 1.0),
    }
    rows, allocs = [], {}
    for name, (s, d, f) in scen.items():
        c, sb, db, note = balance(cost_m.values * f, s, d)
        x, z, _ = solve_linprog(c, sb, db)
        xr = x[:len(wn), :len(sn)]; allocs[name] = xr
        rows.append({"Scenario": name, "Total cost (₹)": round(z, 2), "Units shipped": xr.sum(),
                     "Unmet demand": max(sum(d) - xr.sum(), 0), "Unused stock": max(sum(s) - xr.sum(), 0)})
    df = pd.DataFrame(rows); base_z = df["Total cost (₹)"][0]
    df["Change vs Normal (₹)"] = (df["Total cost (₹)"] - base_z).round(2)
    df["Cost per unit shipped (₹)"] = (df["Total cost (₹)"] / df["Units shipped"]).round(2)
    st.dataframe(df, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(df, x="Scenario", y="Total cost (₹)", text_auto=",.0f", color="Scenario"), width="stretch")
    pick = st.selectbox("Show optimal allocation for", list(allocs))
    st.plotly_chart(px.imshow(plan_df(allocs[pick], wn, sn), text_auto=True, color_continuous_scale="Purples", aspect="auto"), width="stretch")
    st.caption("Total cost covers only shipped units; unmet demand is not costed (assumption). In shortage scenarios compare the cost-per-unit-shipped column, because total cost falls only since fewer units move.")

else:
    st.title("Results & Download")
    st.markdown(f"""
**Optimal cost:** ₹{opt_cost:,.0f}  |  **Baseline (assumed):** ₹{base_cost:,.0f}  |  **Savings:** ₹{savings:,.0f} ({pct:.2f}%)

**Findings**
- Choosing routes by cost instead of habit lowers cost in this illustrative example.
- VAM reached the optimum directly; NWC and Least Cost needed MODI improvement.
- SciPy linprog matches MODI, which validates the hand calculation.

**Limitations:** costs are distance-based only (no tolls, loading charges, time windows); data is illustrative; baseline is assumed. The percentage applies only to this dataset.
""")
    out_df = pd.DataFrame([{"warehouse": wn[i], "store": sn[j], "units": opt_x[i, j], "cost_per_unit": cost_m.iloc[i, j],
                            "route_cost": opt_x[i, j] * cost_m.iloc[i, j]} for i in range(len(wn)) for j in range(len(sn)) if opt_x[i, j] > 0])
    st.dataframe(out_df, hide_index=True, width="stretch")
    st.download_button("Download optimal plan (CSV)", out_df.to_csv(index=False), "optimal_plan.csv", "text/csv")
