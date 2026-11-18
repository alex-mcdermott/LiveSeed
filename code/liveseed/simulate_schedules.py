#!/usr/bin/env python3
"""simulate_schedules.py: offline simulation of LiveSeed exit schedules, best-of-k and preview baselines on a logged pool (App. C).

Why offline: every candidate of a pool is generated to completion with its readings logged, so the pick of
ANY rule, schedule or evaluator is known exactly and its cost is arithmetic.  No GPU is needed here.

Input (--logs; written by liveseed_pool_table.py): one table, CSV or parquet, one row per (prompt, seed):
    prompt_id        str/int
    seed             int
    draw             int (optional)  the seed's rank in the harness's draw order; sub-pools of N take the first N by draw
    E__<block>__<t>  float   normalised entropy reading at block <block>, step <t>   (one column per logged reading)
    P__<eval>__<t>   float   preview score: evaluator <eval> on the predicted clean image decoded at step <t>   (optional)
    S__<eval>        float   score of the finished image under evaluator <eval>  (hpsv21, imagereward, pickscore, hpsv3, vqascore, ...)
Readings are assumed to exist for every step up to the last exit of a schedule (the paper logs steps 1-12).

Outputs (written to --out):
    rows_<model>.csv             one row per rule and pool size: rule, N, nbar (effective generations, N-bar), gain (mean gain of
                                 the pick over the pool mean, in units of --target), lo, hi (95% bootstrap interval over prompts),
                                 frac (fraction of the best-of-N gap closed: mean over prompts of the per-prompt ratio),
                                 frac_ratio_of_means (the same as a ratio of means), win (share of prompts with a positive gain), n
    picks_<model>.csv            the seed every rule picks for every prompt and N
    schedules_<model>.json       what was calibrated in each fold for every schedule: the chosen readings, or the ridge
                                 penalty and its five largest weights
    pareto_<model>.pdf           gain against N-bar for every rule

Usage:
    python liveseed_pool_table.py --run runs/sd14_pool --model sd14 --out runs/pool_sd14.parquet
    python simulate_schedules.py --logs runs/pool_sd14.parquet --model SD14 --T 30 --target S__hpsv21 --out runs/sim_sd14
    python simulate_schedules.py --selftest --out runs/sim_selftest        # runs on synthetic data

Rules implemented
    oracle(N)                           best of N finished candidates under the target
    bestof(k, eval)                     best of k finished candidates under evaluator `eval` (k < N: expectation over random k-subsets)
    preview(step, eval)                 best of N by a preview score at `step`
    psp(schedule, eval)                 Progressive Seed Pruning (PSP): prune on preview scores at 25% and 50% of the steps, finish
                                        the survivors, keep the best by the finished-image evaluator
    liveseed(schedule, form)            prune on the running LiveSeed score (Algorithm A); `form` is single (one reading),
                                        subset5 (LiveSeed-Fast, five readings) or ridge (LiveSeed-Max)
    cascade(s1, keep, step2, eval)      LiveSeed prunes at s1 to `keep`, a preview+evaluator at step2 picks among survivors
A random draw is the reference: every gain is measured against the pool mean.

Conventions
  * All calibrated rules are chosen by two-fold cross-validation over prompts (choose on fold A, score on fold B, swap,
    pool). The folds are interleaved halves: prompt ids sorted as strings, alternate positions.
  * The five-map subset is chosen by running the schedule itself on the calibration prompts (every exit, the survivors
    re-z-scored), not by the gain of a single final-step pick. The ridge is fitted on all readings up to the last exit
    and applied at every exit to the readings available there, as Algorithm A does.
  * Sub-pools of N are the first N seeds in the harness's draw order (`draw` column, else row order), not the N smallest
    seed values.
  * The fraction of the oracle gap is the mean over prompts of the per-prompt ratio (prompts whose oracle gap is zero
    are left out of that mean), with the ratio of means kept in a second column.
"""
import argparse, json, os
import numpy as np
import pandas as pd

# One generator with a fixed seed serves the bootstrap and the random k-subsets, so a given command line is reproducible.
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------ cost model (Sec. 3.4, App. D)
def nfe(N, schedule, T):
    """schedule: list of (step, keep); candidate-steps spent. keep of the last exit is the number finished."""
    cost, alive, prev = 0, N, 0
    for s, k in schedule:
        cost += alive * (s - prev)
        alive, prev = k, s
    return cost + alive * (T - prev)


def nbar(N, schedule, T):
    """Effective number of generations: the candidate-steps of the schedule divided by the T steps of one image."""
    return nfe(N, schedule, T) / T


# ------------------------------------------------------------------------------------------ helpers
def reading_cols(df, max_step):
    """The reading columns E__<block>__<t> with t <= max_step."""
    cols = []
    for c in df.columns:
        if c.startswith("E__"):
            t = int(c.split("__")[2])
            if t <= max_step:
                cols.append(c)
    return cols


def within_pool_z(mat):
    """z-score each column within a pool (rows = candidates of one prompt)."""
    m = mat.mean(0, keepdims=True)
    s = mat.std(0, keepdims=True) + 1e-8
    return (mat - m) / s


def pools(df, prompts, cols, target):
    """yield (prompt, Z[N x R], y[N]) for the given prompts; Z z-scored within the pool."""
    for p in prompts:
        g = df[df.prompt_id == p]
        Z = within_pool_z(np.nan_to_num(g[cols].to_numpy(float)))
        yield p, Z, g[target].to_numpy(float)


def gain_of_pick(score, y):
    """Gain of the top-scored candidate over the pool mean, in the target's units."""
    return y[np.argmax(score)] - y.mean()


def interleaved_folds(prompts):
    """The two folds as [(calibration, test), (test, calibration)]: prompt ids sorted AS STRINGS ("1", "10", "100", ...),
    alternate positions. fit_trajectory_metric.py and seed_select_eval.py use the same convention, so the simulator's
    rows are comparable with theirs."""
    ps = sorted(prompts, key=lambda x: str(x))
    A, B = ps[0::2], ps[1::2]
    return [(A, B), (B, A)]


def first_n(df, N):
    """The first N candidates of every prompt in draw order (row order if there is no `draw` column)."""
    if "draw" in df.columns:
        d = df.sort_values(["prompt_id", "draw"], kind="stable")
    else:
        d = df.assign(_row=np.arange(len(df))).sort_values(["prompt_id", "_row"], kind="stable").drop(columns="_row")
    return d.groupby("prompt_id", group_keys=False, sort=False).head(N).reset_index(drop=True)


# ------------------------------------------------------------------------------------------ rules on one pool
def liveseed_pick(Zfull, cols, exits, rule, return_alive=False):
    """Runs a LiveSeed schedule on one pool and returns the index of the finished candidate.

    Zfull  N x R readings of the pool over ALL logged readings `cols`, z-scored within the pool.
    exits  [(step, keep), ...]. At each exit only the readings of steps <= step are used, and they are z-scored again
           over the survivors, as in Algorithm A; the `keep` best survive.
    rule   dict(kind="ridge", w=weights over cols) or dict(kind="subset", idx=column indices, signs=+1/-1).
    With return_alive, also returns the indices of the survivors of the last exit run, best first.
    """
    alive = np.arange(Zfull.shape[0])
    raw = Zfull
    for s, k in exits:
        avail = [i for i, c in enumerate(cols) if int(c.split("__")[2]) <= s]     # readings taken by step s
        sub = raw[alive][:, avail]
        sub = within_pool_z(sub) if len(alive) > 1 else sub
        if rule["kind"] == "ridge":             # LiveSeed-Max: the weights of the available readings
            w = rule["w"][avail]
            score = sub @ w
        else:                                   # LiveSeed-Fast: the signed sum of the chosen readings available so far
            score = np.zeros(len(alive))
            for j, sg in zip(rule["idx"], rule["signs"]):
                if j in avail:
                    score += sg * sub[:, avail.index(j)]
        order = np.argsort(-score, kind="stable")
        alive = alive[order[:k]]
        if k == 1:
            break
    return (alive[0], alive) if return_alive else alive[0]


def preview_pick(g, step, ev, alive=None):
    """Index of the candidate with the best preview score of evaluator `ev` at `step`, among `alive` (default: all)."""
    col = f"P__{ev}__{step}"
    v = np.nan_to_num(g[col].to_numpy(float), nan=-np.inf)
    idx = np.arange(len(v)) if alive is None else alive
    return idx[np.argmax(v[idx])]


def bestof_expected_gain(y, k, ev_scores, n_sub=200):
    """Expected gain (under the target y) of best-of-k chosen by evaluator scores: the average over n_sub random
    k-subsets of the N candidates, which is what a user who generates k seeds gets on average.

    Returns (gain, index of the pick). For k < N the gain is an expectation and there is no single pick: the index is -1.
    """
    N = len(y)
    if k >= N:
        return y[np.argmax(ev_scores)] - y.mean(), int(np.argmax(ev_scores))
    g = 0.0
    for _ in range(n_sub):
        sub = RNG.choice(N, k, replace=False)
        g += y[sub[np.argmax(ev_scores[sub])]] - y.mean()
    return g / n_sub, -1


# ------------------------------------------------------------------------------------------ calibration (Sec. 3.3)
def greedy_subset(df, prompts, cols, target, exits, max_size=5):
    """LiveSeed-Fast: forward selection of (reading, sign) pairs on the calibration prompts.

    The objective is the mean gain of the pick that the SCHEDULE makes: each candidate subset is run through
    liveseed_pick with the given exits, so a subset is judged as it will be used (every exit, survivors re-z-scored).
    A pair is added only if it raises the objective. Returns (column indices into cols, signs).
    """
    data = list(pools(df, prompts, cols, target))
    chosen, signs = [], []
    best_obj = -np.inf
    for _ in range(max_size):
        best = None
        for j in range(len(cols)):
            if j in chosen:
                continue
            for sg in (+1, -1):
                rule = dict(kind="subset", idx=chosen + [j], signs=signs + [sg])
                obj = np.mean([gain_of_pick_sched(Z, y, cols, exits, rule) for _, Z, y in data])
                if obj > best_obj + 1e-9 and (best is None or obj > best[0]):
                    best = (obj, j, sg)
        if best is None:
            break
        best_obj, j, sg = best
        chosen.append(j)
        signs.append(sg)
    return chosen, signs


def gain_of_pick_sched(Z, y, cols, exits, rule):
    """Gain over the pool mean of the candidate that the schedule finishes under `rule`."""
    return y[liveseed_pick(Z, cols, exits, rule)] - y.mean()


# The grid of ridge penalties. The grid in the code is 1 to 10^4; the paper's rating-prediction results (Sec. 4.4) were
# computed with the grid 10^-1 to 10^3, as the appendix's calibration-settings table states.
RIDGE_ALPHAS = (1, 10, 100, 1000, 10000)


def ridge_fit(df, prompts, cols, target, alphas=RIDGE_ALPHAS):
    """LiveSeed-Max: ridge weights over `cols` on the calibration prompts; returns (weights, chosen alpha).

    Readings are z-scored within each pool and the target is centred within each pool, so the fit has no intercept and
    explains only differences between seeds of one prompt. alpha is chosen by an inner two-fold cross-validation
    (interleaved) on the calibration prompts, scored by the gain of the top-1 pick under all readings of `cols`; the
    ridge is then refitted on all calibration prompts.
    """
    def stack(ps):
        X, Y = [], []
        for _, Z, y in pools(df, ps, cols, target):
            X.append(Z)
            Y.append(y - y.mean())
        return np.vstack(X), np.concatenate(Y)

    inner = interleaved_folds(list(prompts))
    best_a, best_g = None, -np.inf
    for a in alphas:
        g = 0.0
        for A, B in inner:
            X, Y = stack(A)
            w = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ Y)
            g += np.mean([gain_of_pick(Z @ w, y) for _, Z, y in pools(df, B, cols, target)])
        if g > best_g:
            best_g, best_a = g, a
    X, Y = stack(list(prompts))
    return np.linalg.solve(X.T @ X + best_a * np.eye(X.shape[1]), X.T @ Y), best_a


# ------------------------------------------------------------------------------------------ evaluation
def bootstrap_ci(x, n=2000):
    """Mean of x and its 95% percentile bootstrap interval (n resamples): (mean, lo, hi)."""
    x = np.asarray(x, float)
    m = x.mean()
    bs = RNG.choice(x, (n, len(x)), replace=True).mean(1)
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def nearest_preview_step(df, ev, wanted):
    """The latest step <= wanted at which previews of evaluator `ev` exist (the earliest one if none is; None if no previews)."""
    have = sorted({int(c.split("__")[2]) for c in df.columns if c.startswith(f"P__{ev}__")})
    if not have:
        return None
    le = [s for s in have if s <= wanted]
    return max(le) if le else have[0]


def evaluate(df, model, T, target, out, N_list=(10, 20, 50), T_read=12, evals_preview=("imagereward", "hpsv21"),
             preview_steps=(2, 3, 5, 8, 15), forms=("single", "subset5", "ridge"), exits=None):
    """Simulates every rule on the pool `df` for each pool size of N_list and writes the outputs listed in the module docstring.

    T: denoising steps of one image; target: the S__ column the gain is measured in.
    exits: a list of single-exit steps replacing the default schedules (e.g. (1, 2, 3) for a four-step model); None keeps
    the default schedules. Returns the table of rows.
    """
    os.makedirs(out, exist_ok=True)
    prompts = list(df.prompt_id.unique())
    folds = interleaved_folds(prompts)
    N_full = df.groupby("prompt_id").size().min()
    rows, chosen_log, picks = [], {}, []

    def record(name, N, Nb, gains, gaps, pick_by_prompt=None):
        """Adds one row: per-prompt gains and oracle gaps of a rule that spends Nb effective generations."""
        gains = np.asarray(gains, float)
        gaps = np.asarray(gaps, float)
        m, lo, hi = bootstrap_ci(gains)
        ratio = gains / np.where(gaps > 0, gaps, np.nan)
        rows.append(dict(rule=name, N=N, nbar=round(Nb, 2), gain=m, lo=lo, hi=hi, frac=float(np.nanmean(ratio)),
                         frac_ratio_of_means=float(m / gaps.mean()) if gaps.mean() > 0 else np.nan,
                         win=float(np.mean(gains > 0)), n=int(len(gains))))
        if pick_by_prompt:
            for p, seed in pick_by_prompt:
                picks.append(dict(rule=name, N=N, prompt_id=p, seed=seed))

    for N in N_list:
        if N > N_full:
            continue
        dN = first_n(df, N)                                                         # the sub-pool: first N seeds in draw order
        groups = {p: g.reset_index(drop=True) for p, g in dN.groupby("prompt_id", sort=False)}
        order = [p for p in prompts if p in groups]
        y_by_p = {p: groups[p][target].to_numpy(float) for p in order}
        seeds_by_p = {p: groups[p]["seed"].to_numpy() for p in order}
        gaps = [y_by_p[p].max() - y_by_p[p].mean() for p in order]              # oracle gap of each prompt
        record(f"oracle best-of-{N}", N, N, gaps, gaps, [(p, seeds_by_p[p][int(np.argmax(y_by_p[p]))]) for p in order])

        # best-of-k under evaluators (finished images)
        for ev in ("hpsv21", "imagereward", "pickscore", "hpsv3", "vqascore"):
            col = f"S__{ev}"
            if col not in dN.columns:
                continue
            for k in (2, 3, 5, 10):
                if k > N:
                    continue
                res_ = [bestof_expected_gain(y_by_p[p], k, np.nan_to_num(groups[p][col].to_numpy(float), nan=-np.inf)) for p in order]
                record(f"best-of-{k} by {ev} (finished)", N, k, [r[0] for r in res_], gaps,
                       [(p, seeds_by_p[p][r[1]]) for p, r in zip(order, res_) if r[1] >= 0] or None)

        # previews
        for ev in evals_preview:
            for s in preview_steps:
                if f"P__{ev}__{s}" not in dN.columns:
                    continue
                pk = [preview_pick(groups[p], s, ev) for p in order]
                record(f"preview+{ev} @ step {s}, N={N}", N, nbar(N, [(s, 1)], T), [y_by_p[p][i] - y_by_p[p].mean() for p, i in zip(order, pk)],
                       gaps, [(p, seeds_by_p[p][i]) for p, i in zip(order, pk)])
            # PSP at 25% / 50% of the steps, generalised to N: N -> N/2 -> N/4 -> finish the survivors, keep the best by the finished-image evaluator
            s1, s2 = nearest_preview_step(dN, ev, round(0.25 * T)), nearest_preview_step(dN, ev, round(0.5 * T))
            if s1 is not None and s2 is not None and s2 > s1 and f"S__{ev}" in dN.columns:
                psp_sched = [(s1, max(N // 2, 1)), (s2, max(N // 4, 1))]
                gains, pk = [], []
                for p in order:
                    g = groups[p]
                    alive = np.arange(len(g))
                    for s, k in psp_sched:
                        v = np.nan_to_num(g[f"P__{ev}__{s}"].to_numpy(float), nan=-np.inf)[alive]
                        alive = alive[np.argsort(-v, kind="stable")[:k]]
                    v = np.nan_to_num(g[f"S__{ev}"].to_numpy(float), nan=-np.inf)[alive]
                    pick = alive[np.argmax(v)]
                    gains.append(y_by_p[p][pick] - y_by_p[p].mean())
                    pk.append((p, seeds_by_p[p][pick]))
                record(f"PSP {N}->{N // 2}->{N // 4} by {ev} previews @ {s1},{s2}", N, nfe(N, psp_sched, T) / T, gains, gaps, pk)

        # LiveSeed schedules (cross-validated, schedule-aware calibration)
        schedules = {"exit3": [(3, 1)], "exit6": [(6, 1)], "exit12": [(12, 1)],
                     "two-exit(3,N/2)(8,1)": [(3, max(N // 2, 1)), (8, 1)],
                     "two-exit(2,N/2)(6,1)": [(2, max(N // 2, 1)), (6, 1)],
                     "three-exit(2,N/2)(5,N/5)(10,1)": [(2, max(N // 2, 1)), (5, max(N // 5, 1)), (10, 1)]}
        if exits is not None:                                                      # only the requested single exits
            schedules = {f"exit{s}": [(s, 1)] for s in exits}
        for sname, sched in schedules.items():
            cap = sched[-1][0]                      # the last exit: later readings are never used
            cols_cap = reading_cols(dN, cap)
            if not cols_cap:
                continue
            for form in forms:
                gains, pk = [], []
                for fi, (cal, test) in enumerate(folds):
                    # calibrate on one fold ...
                    if form == "ridge":
                        w, a = ridge_fit(dN, cal, cols_cap, target)
                        rule = dict(kind="ridge", w=w)
                        chosen_log[f"N{N}|{sname}|{form}|fold{fi}"] = dict(
                            alpha=a, top=[(cols_cap[i], float(w[i])) for i in np.argsort(-np.abs(w))[:5]])
                    else:
                        idx, signs = greedy_subset(dN, cal, cols_cap, target, sched, max_size=1 if form == "single" else 5)
                        rule = dict(kind="subset", idx=idx, signs=signs)
                        chosen_log[f"N{N}|{sname}|{form}|fold{fi}"] = [(cols_cap[j], int(s)) for j, s in zip(idx, signs)]
                    # ... and run the schedule on the other fold's pools
                    for p, Z, y in pools(dN, test, cols_cap, target):
                        pick = liveseed_pick(Z, cols_cap, sched, rule)
                        gains.append(y[pick] - y.mean())
                        pk.append((p, seeds_by_p[p][pick]))
                gp = [y_by_p[p].max() - y_by_p[p].mean() for p, _ in pk]
                record(f"LiveSeed {form} {sname}, N={N}", N, nbar(N, sched, T), gains, gp, pk)

        # cascade: LiveSeed-Fast prunes the pool to 30% at step 3 at no extra cost; a preview scored by the evaluator at the
        # first available of steps 6, 5, 8 then picks among the survivors
        for ev in evals_preview:
            s2 = next((s for s in (6, 5, 8) if f"P__{ev}__{s}" in dN.columns), None)
            if s2 is None:
                continue
            keep = max(int(round(0.3 * N)), 1)
            cols_cap = reading_cols(dN, 3)
            gains, pk = [], []
            for cal, test in folds:
                # the five maps of the first stage are calibrated as for the single exit at step 3
                idx, signs = greedy_subset(dN, cal, cols_cap, target, [(3, 1)], max_size=5)
                rule = dict(kind="subset", idx=idx, signs=signs)
                for p, Z, y in pools(dN, test, cols_cap, target):
                    _, alive = liveseed_pick(Z, cols_cap, [(3, keep)], rule, return_alive=True)
                    pick = preview_pick(groups[p], s2, ev, alive)
                    gains.append(y[pick] - y.mean())
                    pk.append((p, seeds_by_p[p][pick]))
            gp = [y_by_p[p].max() - y_by_p[p].mean() for p, _ in pk]
            record(f"cascade LiveSeed@3 keep {keep} -> preview+{ev}@{s2}, N={N}", N, nbar(N, [(3, keep), (s2, 1)], T), gains, gp, pk)

        # adaptive budget: does the spread of the readings at steps <= 3 predict the oracle gap of the prompt?
        cols3 = reading_cols(dN, 3)
        if cols3:
            from scipy.stats import spearmanr
            spread = [np.nan_to_num(groups[p][cols3].to_numpy(float)).std(0).mean() for p in order]
            rho = spearmanr(spread, gaps).correlation
            chosen_log[f"N{N}|spread-vs-gap spearman"] = float(rho)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, f"rows_{model}.csv"), index=False)
    pd.DataFrame(picks).to_csv(os.path.join(out, f"picks_{model}.csv"), index=False)
    json.dump(chosen_log, open(os.path.join(out, f"schedules_{model}.json"), "w"), indent=1, default=str)
    plot_pareto(res, model, out)
    return res


def plot_pareto(res, model, out):
    """Plots the gain of every rule against its effective generations, for the largest pool size simulated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    def sel(prefix, N=None):
        r = res[res.rule.str.startswith(prefix)]
        if N is not None:
            r = r[r.N == N]
        return r.sort_values("nbar")

    Nmax = int(res.N.max())
    ls = sel("LiveSeed ridge", Nmax)
    ax.plot(ls.nbar, 1000 * ls.gain, "o", color="#2a78d6", ms=3.5, label=f"LiveSeed-Max, all schedules (N={Nmax})")
    ls5 = sel("LiveSeed subset5", Nmax)
    ax.plot(ls5.nbar, 1000 * ls5.gain, "o", mfc="white", color="#2a78d6", ms=3.5, label="LiveSeed-Fast (five maps)")
    pv = sel("preview+imagereward", Nmax)
    ax.plot(pv.nbar, 1000 * pv.gain, "^", color="#1baf7a", ms=3.5, label="preview + ImageReward")
    pv2 = sel("preview+hpsv21", Nmax)
    ax.plot(pv2.nbar, 1000 * pv2.gain, "^", mfc="white", color="#1baf7a", ms=3.5, label="preview + HPSv2.1 (verifier = metric)")
    ps = sel("PSP", Nmax)
    ax.plot(ps.nbar, 1000 * ps.gain, "s", color="#1baf7a", ms=3.5, label="PSP")
    for ev, c, m in (("hpsv21", "#8a8985", "--"), ("imagereward", "#8a8985", ":")):
        b = sel("best-of-", Nmax)
        b = b[b.rule.str.contains(ev)]
        if len(b):
            ax.plot(b.nbar, 1000 * b.gain, m, color=c, lw=0.9, label=f"best-of-k by {ev} (finished)")
    ca = sel("cascade", Nmax)
    ax.plot(ca.nbar, 1000 * ca.gain, "D", color="#4a3aa7", ms=3.5, label="cascade")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xticklabels(["1", "2", "3", "5", "10"])
    ax.set_xlabel("effective generations $\\bar N$")
    ax.set_ylabel("gain of the pick over a random draw\n(target units $\\times 10^{-3}$)")
    ax.set_title(model, loc="left", fontsize=8)
    ax.legend(fontsize=5, loc="lower right")
    ax.grid(color="#e6e4dd", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"pareto_{model}.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------ synthetic self-test
def synthetic(n_prompts=60, N=20, T=30, T_read=12, n_blocks=6, seed=0):
    """toy pools: a latent 'quality' q per candidate; readings = signal(q) at some blocks/steps + noise; previews get sharper with step."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prompts):
        base = rng.normal(0, 1)                 # the prompt's difficulty, shared by its candidates
        for s in range(N):
            q = rng.normal(0, 1)
            row = dict(prompt_id=p, seed=int(rng.integers(1, 2**31 - 1)), draw=s)
            for b in range(n_blocks):
                for t in range(1, T_read + 1):
                    sig = 0.6 * q if (b in (1, 4) and 2 <= t <= 9) else 0.0     # only blocks 1 and 4 carry signal, at steps 2-9
                    row[f"E__b{b}__{t}"] = 0.9 + 0.02 * (sig + rng.normal(0, 1))
            for ev, w in (("imagereward", 1.0), ("hpsv21", 1.0)):
                for t in (2, 3, 5, 8, 15):
                    sharp = (min(t, 15) / 15) ** 1.5
                    row[f"P__{ev}__{t}"] = sharp * q + (1 - sharp) * rng.normal(0, 1)
            row["S__hpsv21"] = 0.27 + 0.01 * (base + q) + 0.003 * rng.normal()
            row["S__imagereward"] = base + 0.8 * q + 0.5 * rng.normal()
            row["S__pickscore"] = 21 + base + 0.6 * q + 0.6 * rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def selftest(out):
    """Checks three conventions of the module docstring on synthetic pools, then runs the whole evaluation; raises on failure."""
    df = synthetic()
    # the sub-pool of N is the first N by draw, whatever the seed values
    d10 = first_n(df, 10)
    assert (d10.groupby("prompt_id")["draw"].max() == 9).all(), "sub-pool is not the first N by draw order"
    # interleaved folds
    A, B = interleaved_folds(list(df.prompt_id.unique()))[0]
    assert set(A).isdisjoint(B) and len(A) + len(B) == df.prompt_id.nunique() and A[0] == 0 and B[0] == 1, "folds are not interleaved"
    # a schedule-aware subset must do at least as well on its own schedule as the final-step subset applied to it (on the calibration prompts)
    cols = reading_cols(d10, 8)
    sched = [(3, 5), (8, 1)]
    cal = A
    idx, signs = greedy_subset(d10, cal, cols, "S__hpsv21", sched, 5)
    idx0, signs0 = greedy_subset(d10, cal, cols, "S__hpsv21", [(8, 1)], 5)
    g1 = np.mean([gain_of_pick_sched(Z, y, cols, sched, dict(kind="subset", idx=idx, signs=signs))
                  for _, Z, y in pools(d10, cal, cols, "S__hpsv21")])
    g0 = np.mean([gain_of_pick_sched(Z, y, cols, sched, dict(kind="subset", idx=idx0, signs=signs0))
                  for _, Z, y in pools(d10, cal, cols, "S__hpsv21")])
    assert g1 >= g0 - 1e-9, f"schedule-aware calibration ({g1:.5f}) below the final-step one ({g0:.5f}) on its own schedule"
    res = evaluate(df, "synthetic", 30, "S__hpsv21", out, N_list=(10, 20))
    assert os.path.exists(os.path.join(out, "picks_synthetic.csv"))
    print(res[["rule", "N", "nbar", "gain", "lo", "hi", "frac", "frac_ratio_of_means", "win"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"[selftest] ok: schedule-aware subset {g1:+.5f} vs final-step subset {g0:+.5f} on the (3,5),(8,1) schedule, calibration prompts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs")                               # the pool table written by liveseed_pool_table.py (CSV or parquet)
    ap.add_argument("--model", default="model")             # a label for the output file names
    ap.add_argument("--T", type=int, default=30)            # denoising steps of one image
    ap.add_argument("--target", default="S__hpsv21")        # the finished-image score the gain is measured in and calibrated on
    ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--N", default="10,20,50", help="pool sizes to simulate (each the first N seeds in draw order)")
    ap.add_argument("--forms", default="single,subset5,ridge")
    ap.add_argument("--exits", default=None,
                    help="comma list of single-exit steps replacing the default schedules (for example 1,2,3 on a four-step model)")
    ap.add_argument("--preview-steps", default="2,3,5,8,15", help="preview steps to evaluate (for example 1,2,3 on a four-step model)")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.out)
    else:
        df = pd.read_parquet(a.logs) if a.logs.endswith(".parquet") else pd.read_csv(a.logs)
        res = evaluate(df, a.model, a.T, a.target, a.out, N_list=tuple(int(x) for x in a.N.split(",")), forms=tuple(a.forms.split(",")),
                       preview_steps=tuple(int(x) for x in a.preview_steps.split(",")),
                       exits=tuple(int(x) for x in a.exits.split(",")) if a.exits else None)
        print(res.to_string(index=False))
