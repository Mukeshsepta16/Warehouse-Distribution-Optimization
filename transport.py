"""Transportation-problem methods: NWC, Least Cost, VAM, MODI, and SciPy linprog."""
import numpy as np
from scipy.optimize import linprog


def total_cost(cost, x):
    return float((np.asarray(cost) * np.asarray(x)).sum())


def balance(cost, supply, demand):
    """Add a dummy row/column (cost 0) if total supply != total demand."""
    cost = np.array(cost, float); supply = list(map(float, supply)); demand = list(map(float, demand))
    note = "Balanced: no dummy needed."
    diff = sum(supply) - sum(demand)
    if diff > 1e-9:
        cost = np.hstack([cost, np.zeros((cost.shape[0], 1))]); demand.append(diff)
        note = f"Supply exceeds demand by {diff:g}: dummy destination added (unused stock)."
    elif diff < -1e-9:
        cost = np.vstack([cost, np.zeros((1, cost.shape[1]))]); supply.append(-diff)
        note = f"Demand exceeds supply by {-diff:g}: dummy warehouse added (unmet demand)."
    return cost, supply, demand, note


def north_west(cost, supply, demand):
    s, d = list(supply), list(demand); m, n = len(s), len(d)
    x = np.zeros((m, n)); i = j = 0
    while i < m and j < n:
        q = min(s[i], d[j]); x[i, j] = q; s[i] -= q; d[j] -= q
        if s[i] == 0 and i < m - 1: i += 1
        elif d[j] == 0: j += 1
        else: i += 1
    return x


def least_cost(cost, supply, demand):
    c = np.array(cost, float); s, d = list(supply), list(demand); m, n = c.shape
    x = np.zeros((m, n)); order = sorted(((c[i, j], i, j) for i in range(m) for j in range(n)))
    for _, i, j in order:
        q = min(s[i], d[j])
        if q > 0:
            x[i, j] += q; s[i] -= q; d[j] -= q
    return x


def vam(cost, supply, demand):
    c = np.array(cost, float); s, d = list(supply), list(demand); m, n = c.shape
    x = np.zeros((m, n)); rows, cols = set(range(m)), set(range(n))
    def pen(vals):
        vals = sorted(vals)
        return vals[1] - vals[0] if len(vals) > 1 else vals[0]
    while rows and cols:
        rp = {i: pen([c[i, j] for j in cols]) for i in rows}
        cp = {j: pen([c[i, j] for i in rows]) for j in cols}
        bi = max(rp, key=rp.get); bj = max(cp, key=cp.get)
        if rp[bi] >= cp[bj]:
            i = bi; j = min(cols, key=lambda jj: c[i, jj])
        else:
            j = bj; i = min(rows, key=lambda ii: c[ii, j])
        q = min(s[i], d[j]); x[i, j] += q; s[i] -= q; d[j] -= q
        if s[i] == 0 and (d[j] != 0 or len(rows) > 1): rows.discard(i)
        else: cols.discard(j)
    return x


def _basis(x):
    """Cells with allocation > 0, completed to m+n-1 cells (zero cells) without forming a cycle."""
    m, n = x.shape
    parent = list(range(m + n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    basis = set()
    for i in range(m):
        for j in range(n):
            if x[i, j] > 1e-9:
                basis.add((i, j)); parent[find(i)] = find(m + j)
    for i in range(m):
        for j in range(n):
            if len(basis) >= m + n - 1: return basis
            if (i, j) not in basis and find(i) != find(m + j):
                basis.add((i, j)); parent[find(i)] = find(m + j)
    return basis


def _cycle(cells, start):
    cells = set(cells)
    def dfs(path, by_row):
        i, j = path[-1]
        cands = [(i, b) for (a, b) in cells if a == i and b != j] if by_row else \
                [(a, j) for (a, b) in cells if b == j and a != i]
        for cnd in cands:
            if cnd == start and len(path) >= 4: return path
            if cnd not in path:
                r = dfs(path + [cnd], not by_row)
                if r: return r
        return None
    return dfs([start], True)


def modi(cost, x0):
    """Run MODI until optimal. Returns final x and a log of each iteration.
    The basis is kept explicitly so degenerate (zero-shift) pivots still make progress."""
    c = np.array(cost, float); x = np.array(x0, float).copy(); m, n = c.shape; log = []
    basis = _basis(x)
    for _ in range(200):
        u = [None] * m; v = [None] * n; u[0] = 0.0
        changed = True
        while changed:
            changed = False
            for (i, j) in basis:
                if u[i] is not None and v[j] is None: v[j] = c[i, j] - u[i]; changed = True
                elif v[j] is not None and u[i] is None: u[i] = c[i, j] - v[j]; changed = True
        d = np.full((m, n), np.nan)
        for i in range(m):
            for j in range(n):
                if (i, j) not in basis: d[i, j] = c[i, j] - u[i] - v[j]
        (ei, ej) = np.unravel_index(np.nanargmin(d), d.shape); dmin = d[ei, ej]
        log.append({"u": [float(a) for a in u], "v": [float(a) for a in v], "d": d.copy(),
                    "cost": total_cost(c, x), "entering": (int(ei), int(ej)), "dmin": float(dmin)})
        if dmin >= -1e-9: break
        cyc = _cycle(basis | {(ei, ej)}, (ei, ej))
        minus = [p for k, p in enumerate(cyc) if k % 2 == 1]
        theta = min(x[p] for p in minus)
        leaving = next(p for p in minus if x[p] == theta)
        for k, p in enumerate(cyc): x[p] += theta if k % 2 == 0 else -theta
        basis.add((ei, ej)); basis.discard(leaving)
    return x, log


def solve_linprog(cost, supply, demand):
    c = np.array(cost, float); m, n = c.shape
    A_ub = np.zeros((m, m * n)); A_eq = np.zeros((n, m * n))
    for i in range(m): A_ub[i, i * n:(i + 1) * n] = 1
    for j in range(n): A_eq[j, j::n] = 1
    r = linprog(c.flatten(), A_ub=A_ub, b_ub=supply, A_eq=A_eq, b_eq=demand,
                bounds=(0, None), method="highs")
    return r.x.reshape(m, n).round(6), r.fun, r.message


def run_all(cost, supply, demand):
    """Balance, run all methods, return dict of results."""
    c, s, d, note = balance(cost, supply, demand)
    out = {"note": note, "cost_bal": c, "supply_bal": s, "demand_bal": d}
    out["NWC"] = north_west(c, s, d); out["LCM"] = least_cost(c, s, d); out["VAM"] = vam(c, s, d)
    out["MODI_from_NWC"], out["modi_log"] = modi(c, out["NWC"])
    out["LP"], out["lp_cost"], out["lp_msg"] = solve_linprog(c, s, d)
    return out
