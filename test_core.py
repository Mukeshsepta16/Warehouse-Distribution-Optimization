import numpy as np
from transport import *
cost = np.array([[6,14,15,26],[19,28,20,18],[30,35,35,11]]); S=[300,250,200]; D=[150,180,200,220]
r = run_all(cost,S,D)
for k in ["NWC","LCM","VAM","MODI_from_NWC","LP"]:
    print(k, total_cost(r["cost_bal"], r[k]))
print(r["NWC"]); print(r["VAM"]); print(r["MODI_from_NWC"].round()); print(r["LP"].round())
print("MODI iterations:", len(r["modi_log"]))
for l in r["modi_log"]: print(l["u"], l["v"], l["dmin"], l["cost"])
base = 12450; print("savings", base-r["lp_cost"], round((base-r["lp_cost"])/base*100,2))
# unbalanced
for name,(s,d) in {"short":([300,250,0],D), "surplus":([400,300,200],D)}.items():
    rr = run_all(cost,s,d); print(name, rr["note"], total_cost(rr["cost_bal"], rr["LP"]), total_cost(rr["cost_bal"], rr["MODI_from_NWC"]))
