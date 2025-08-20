# Dynamic Fantasy Draft Optimizer — Technical Spec (High-Level with Algorithmic Anchors)

## 1) Purpose & Design Philosophy
The draft system’s goal is to **maximize expected starter lineup points under uncertainty** in a snake draft. Unlike static VOR-based strategies, it leverages **probabilistic simulation, opponent modeling, and scenario-driven optimization** to dynamically evaluate whether to draft a player now or risk waiting until the next turn.

Principle: **No hard-coded heuristics.** Decisions come from consistent probabilistic/statistical reasoning aligned with maximizing lineup value.

---

## 2) Data Sources & Structures
- **CSV Input (e.g., ESPN projections):** Provides `player_id`, `name`, `position`, `overall_rank`, and optionally `proj`. Used to initialize player pools.
- **Projection Distribution:** Each player is represented not by a point estimate but by a **distribution of outcomes** (e.g., Beta-PERT samples). This distribution captures ceiling, floor, and uncertainty.
- **League Config:** JSON/dict specifying teams, starter slots, roster sizes, and scoring rules.
- **Draft Order:** Precomputed serpentine order list mapping global picks → team indices.

---

## 3) Core Components & Functions
### 3.1 Player Uncertainty Modeling
- **Function:** `generate_samples(player)`
- **Method:** Beta-PERT or similar, using projection ±20% envelope (configurable). Produces per-player sample arrays.
- **Rationale:** Keeps player valuation scenario-based, not deterministic.

### 3.2 Draft Pool Advancement
- **Function:** `advance_pool(pool, pick_from, pick_to, pick_prob_fn)`
- **Method:** Sequentially simulate opponent picks between two global picks.
- **Statistical Method:** Sampling opponents using `pick_prob_fn` (softmax over projection, ADP model, or learned probabilities).
- **Goal:** Ensure candidate availability reflects draft context.

### 3.3 Best Starter Sum (Lineup Solver)
- **Function:** `best_starter_sum(roster, pool, draw_idx)`
- **Method:** Greedy optimizer:
  1. Select best QB/RB/WR/TE per slots in this draw.
  2. Select best RB/WR/TE for FLEX.
- **Statistical Method:** Uses scenario-specific samples; repeated across draws.
- **Goal:** Provides a robust “starter lineup value” per draw.

### 3.4 Marginal Starter Gain (MSG)
- **Function:** `marginal_starter_gain(candidate, roster, pool)`
- **Method:** Compare lineup value with vs. without candidate in the same draw.
- **Statistical Anchor:** Expected value across multiple draws.
- **Purpose:** Captures how much a player tangibly improves the starter lineup.

### 3.5 Opportunity Cost (OC)
- **Function:** `opportunity_cost(roster, pool, current_pick, next_pick)`
- **Method:** Difference between current starter lineup value and expected lineup value at next pick.
- **Statistical Method:** Monte Carlo simulation of opponent picks between now and next pick.
- **Purpose:** Quantifies the cost of waiting.

### 3.6 Candidate Scoring
- **Function:** `score_candidates(pool, roster, current_pick, next_pick)`
- **Score Formula:** `E[MSG] – E[OC]`
- **Rationale:** Pick player with highest net contribution under uncertainty.
- **Debug Outputs:** Logs MSG, OC, survival probabilities, starter inclusion rates.

### 3.7 Beam/Horizon Search (Optional Extension)
- **Function:** `beam_plan(horizon, beam_width, pool, roster)`
- **Method:** Simulate sequences of picks over H future turns, keeping top-B by expected lineup value.
- **Statistical Anchor:** Tree search with Monte Carlo rollouts.
- **Purpose:** Incorporates longer-term planning beyond single-turn optimization.

---

## 4) Probability Models
- **Opponent Pick Probability (default):** Softmax over mean projection with configurable temperature.
- **Deterministic Mode:** Always pick best available by rank (for sanity tests).
- **Extensions:** Incorporate ADP distributions, positional demand, or machine-learned draft tendencies.

---

## 5) Simulation & Execution Flow
1. **Initialize:** Load players, generate projection samples, build serpentine order.
2. **At each pick:**
   - Advance pool from last pick to current pick.
   - Compute `B_now` (starter sum baseline).
   - Compute `B_next` (starter sum at next pick after simulating opponent picks).
   - For each candidate:
     - Compute `MSG` = improvement in `B_now` if picked now.
     - Compute `OC` = expected loss from waiting.
     - Compute `Score = E[MSG] – E[OC]`.
   - Select best candidate.
   - Mutate pool (remove selected player).
3. **Repeat until roster complete.**

---

## 6) Telemetry & Diagnostics
- **Survival Probabilities:** For each candidate, % chance they survive to next pick.
- **Starter Inclusion Rates:** % of draws where candidate cracks the starter lineup.
- **Score Breakdown:** `E[MSG]`, `E[OC]`, `E[Score]` for top-K.
- **Sanity Checks:**
  - ESPN deterministic mode → Top-4 gone at 1.05.
  - No duplicate picks.
  - Serpentine order invariants.

---

## 7) Research & Experimentation Levers
- **Uncertainty Models:** Beta-PERT, lognormal, empirical residuals.
- **Opponent Models:** From softmax(rank) → ADP tables → ML sequence models.
- **Objective Variants:** Optimize median, floor (p25), or ceiling (p75) lineup scores.
- **Horizon Depth:** Adjust H (2–4 turns) for strategic foresight.

---

## 8) Roadmap
- **Phase 1:** MVP engine (MSG – OC per pick), deterministic opponent, logs.
- **Phase 2:** Probabilistic opponents, risk modes, deeper telemetry.
- **Phase 3:** Horizon/beam planning, richer opponent modeling, backtesting.

---

## 9) North Star Principle
Every piece of logic should trace back to:
**“Does this increase expected starter lineup points under uncertainty, relative to waiting?”**

- VOR is not discarded but reframed: it’s a **diagnostic metric**, not the optimization objective.
- MSG – OC is the unifying decision rule.
- Horizon/beam search provides the long-term planning extension.

---



---

# Technical Design Deep‑Dive (System‑Level)

> Goal: Keep it **dynamic, lineup‑aware, probabilistic**—no hard caps—while being explicit about modules, function entry points, and the statistical methods we’ll use at each stage. Function names are prescriptive; internals are intentionally method‑agnostic (we’ll test variants).

## A. System Boundaries & Assumptions
- **Draft Type:** Snake drafts, single QB, standard FLEX (RB/WR/TE). K/DST supported but not privileged.
- **Objective:** Maximize expected **Starter Sum** (SS) across uncertainty, using **MSG − OC** as the decision criterion.
- **Scope:** Draft‑day only (no waivers/trades). Bench value considered only insofar as it impacts future SS via OC/horizon.

## B. Data Contracts (Sources & Schemas)
- **Primary table:** `/mnt/data/espn_projections_20250814.csv` (authoritative board for ranks and/or projections).
  - Expected columns (best‑effort mapping):
    - `player_id` (or derived from name+team+pos), `player_name`, `position` in {QB,RB,WR,TE,K,DST}
    - `overall_rank` (int); optional `proj_points` (float); optional `team`, `bye`, `adp`.
  - If `proj_points` missing → derive `baseline_proj` by a **rank→points heuristic** (pos‑aware decay). Replaceable later with model‑based projections.
- **Config file (YAML/JSON):** league settings, scoring, round count, horizon/beam knobs, opponent model selection.
- **Optional feeds (later phases):** historical ADP curves, injury tags, team pass/run rates. All optional; the engine must run on CSV alone.

### Loaders (module: `data_io.py`)
- `load_projections(path) -> DataFrame` (validates columns, casts numerics)
- `build_player_table(df, scoring_cfg) -> PlayerTable` (resolves IDs, attaches scoring, fallback projections if needed)

## C. Core Runtime Objects
- `PlayerTable`: immutable table (arrays) for fast lookups: ids, positions, base projections, per‑player samples.
- `DraftState`: canonical remaining pool (list of indices), roster map `{pos -> [ids]}`, global pick index, RNG seed.
- `LeagueConfig`: `n_teams`, `starters_by_pos`, `rounds`, `scoring`.
- `OpponentModel`: strategy object exposing `pick_prob(pool, global_pick, context) -> probs`.

## D. Statistical & Probabilistic Methods (by stage)
> We’ll specify **the class of methods**, not fixed formulas—so we can swap variants during experiments.

### D1) Player Outcome Uncertainty
- **Method family:** Parametric distributions per player (e.g., **PERT**, log‑normal, or empirical sampling) around baseline projection.
- **Sampling API:**
  - `sample_player_outcomes(player_table, n_draws, seed) -> Samples[N_players, N_draws]`
  - Correlation (later): pos‑level or team‑level factors as a low‑rank perturbation (toggleable).

### D2) Opponent Pick Behavior & Survival
- **Baselines:**
  1) **Deterministic ESPN order** (top remaining by `overall_rank`).
  2) **Rank‑weighted softmax** on a utility proxy (e.g., projection or blended rank).
- **Contextual variant (Phase 2):** condition probs on draft context (team needs, positional runs, ADP deltas) using a calibrated classifier.
- **API:** `pick_prob_fn(pool_view, global_pick, context) -> probs[len(pool_view)]`
- **Survival/Advance:** sequential sampling to consume the correct number of opponent picks; no closed‑form survival shortcuts.

### D3) Starter Sum (SS) Solver per Draw
- **Method:** Greedy selection per position plus FLEX maximization is used (provably optimal under independent per‑draw values and single FLEX).
- **API:** `best_starter_sum(roster, pool_view, draw_idx, league_cfg, samples) -> float`
- **Extension:** multiple FLEX supported by repeating the FLEX step; Superflex modeled by allowing QB into FLEX set.

### D4) Opportunity Cost (OC) & Marginal Starter Gain (MSG)
- **OC:** For each draw, compute `SS(now)` from the advanced current pool, then simulate opponents to next pick and compute `SS(next)`. OC draw = `SS(now) − SS(next)`.
- **MSG(c):** Insert candidate `c`, recompute `SS(now_with_c)`; MSG draw = `SS(now_with_c) − SS(now)`.
- **Risk modes:** mean/median/quantile expectation; optional downside‑weighted aggregator.

### D5) Decision Score
- **Score(c):** Expectation of `MSG(c) − OC` over draws.
- **Tie‑breaks:** survival margin, roster balance proxies, or variance preference based on risk setting.

## E. Algorithms & Orchestration

### E1) Serpentine Engine (module: `draft_order.py`)
- `serpentine_order(n_teams, rounds) -> List[int]`
- `next_pick_indices(order, team_id, current_index) -> (curr_idx, next_idx)`
- Invariants: round‑1 ascending, round‑2 descending; team indices consistent with global picks.

### E2) Pool Advancement (module: `advance.py`)
- `advance_pool(pool, from_pick, to_pick, pick_prob_fn, rng, context) -> pool'`
- Accepts a *view* of the pool (list of indices). Mutates a copy when simulating; the canonical pool is mutated only when a pick is committed.

### E3) Candidate Scoring (module: `engine.py`)
- `score_candidates(pool, roster, current_pick, next_pick, player_table, samples, league_cfg, opponent_model, n_draws, top_k) -> List[ScoredCandidate]`
- Steps inside:
  1) Build **pool_at_current** by advancing from last committed pick to `current_pick` (once per decision; cached). 
  2) Precompute `SS(now)` and `SS(next)` per draw (cacheable across candidates). 
  3) Select top‑K candidates by median/mean; for each, compute MSG via re‑solve. 
  4) Return sorted list with breakdown: `E[MSG]`, `E[OC]`, `E[Score]`, `%StarterInclusion` (fraction of draws where candidate enters starters/FLEX).

### E4) Recommendation API (module: `recommend.py`)
- `recommend_pick(...) -> Recommendation{player_id, explanation, debug}`
- Explanation includes survival to next pick, SS diffs, and rationale.

## F. Horizon / Beam Extension (optional module: `horizon.py`)
- **Problem:** Choose a sequence over our next H turns maximizing expected SS.
- **Search:** Beam search with width B, branching on top‑M candidates at each of *our* turns; opponents are always advanced in between using the same opponent model.
- **State:** `(pool_sig, roster_sig, draw_idx, depth)`; reuse SS and advance caches.
- **APIs:**
  - `plan_horizon(pool, roster, current_pick, order, H, B, M, ...) -> Plan{seq, value}`
  - `apply_plan_step(plan, actual_pick_result) -> updated_plan` (gracefully re‑plan on drift).

## G. Caching & Signatures (module: `cache.py`)
- **Pool signature:** top‑N ids by current utility proxy (e.g., median draw value) → tuple.
- **Roster signature:** sorted ids by slot; include position counts.
- **Cached functions:** `SS(...)`, `advance_pool(...)`, `expected_best_at_pick(...)` if used.
- **Invalidation:** on any committed pick; otherwise shared within a decision.

## H. Performance Targets & Knobs
- **Targets:** <2s early rounds (300 draws, K≈32), <1s later.
- **Knobs:** `n_draws_by_round`, `top_k_by_round`, beam width/branching, signature top‑N size.
- **Implementation notes:** use NumPy arrays for samples; pre‑index players by position; avoid Python object churn in inner loops.

## I. Telemetry, Debug, and Invariants (module: `telemetry.py`)
- **Debug prints (guarded):**
  - First 10 opponent picks at our first turn (prove ESPN #1–#4 gone at 1.05 in deterministic mode).
  - Top‑5 candidates with `E[MSG]`, `E[OC]`, `%StarterInclusion`, `E[Score]`.
- **Invariants:**
  - No duplicate picks; pool size decrements by one per pick.
  - Deterministic mode survival sanity: `P(survive(ESPN#1, picks_before_1.05)) = 0`.
  - Serpentine order integrity.

## J. Configuration & Experimentation (module: `config.py`)
- **Runtime config:** opponent model choice (+ temperature), draw counts, candidate K, risk mode, horizon/beam knobs.
- **League config:** teams, starters_by_pos, FLEX rules, rounds, scoring settings.
- **Repro:** global RNG seed; per‑decision sub‑seeds to stabilize UIs.

## K. Evaluation & Backtests (module: `eval.py`)
- **Draft simulations:** replay drafts under fixed opponent models; compare engines: static VOR, MSG only, MSG−OC, horizon.
- **Metrics:** expected SS at each round, regret vs oracle (perfect foresight), survival calibration, decision latency, stability by pick slot.
- **Ablations:** remove OC, remove FLEX, deterministic vs stochastic opponents.

## L. Integration with Existing Code
- **Current artifacts:** `/mnt/data/starter_optimizer.py` (source of initial logic), `/mnt/data/espn_projections_20250814.csv` (board).
- **Plan:**
  - Extract production‑ready pieces into modules above.
  - Replace any proj‑only scoring with `score_candidates(...)` pipeline.
  - Keep original VOR outputs as diagnostics in telemetry (not as the objective).

## M. Failure Modes & Mitigations
- **Opponent model too “soft” → unrealistic survivals:** provide deterministic fallback & expose temperature.
- **Rank string sorting errors:** enforce numeric casts at load.
- **Missing projections:** activate rank→points heuristic; warn visibly.
- **Performance spikes:** adaptive draw count based on round and pool size; use caches.

## N. Roadmap (Incremental)
1) **MVP:** MSG−OC decision engine; deterministic opponent; SS solver; caching; invariants; telemetry.
2) **Phase 2:** Stochastic/learned opponent; risk modes; richer debug UX; backtests.
3) **Phase 3:** Horizon/beam; correlation modeling; UI polish; plug‑and‑play data sources.

## O. Open Questions
- Do we incorporate inter‑player correlations now (team/bye/stacking) or keep for Phase 3?
- What’s the right default risk mode (mean vs median) per league size?
- How to weight bench value beyond OC (e.g., injury insurance) without diluting the starter‑first objective?

