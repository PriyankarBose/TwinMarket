# Reuse Guide: Building a Social Cryptocurrency Trading System from TwinMarket

## 1) Core orchestrator

### `simulation.py`
- `init_simulation(...)`: main daily simulation loop (load users, graph, beliefs, run agents, persist outputs, match trades, update forum scores).
- `process_user_input(...)`: per-user pipeline wrapper that prepares profile + belief + strategy context and invokes the trading agent.

**Why reuse:** this file gives a scalable event loop for multi-agent simulation and can be adapted from stock sessions to 24/7 crypto sessions by replacing trading-day checks and symbol universes.

## 2) Agent decision and social cognition

### `trader/trading_agent.py`
- `PersonalizedStockTrader`: central class containing user state, social context, market/news context, and decision logic.
- `input_info(...)`: end-to-end per-user reasoning cycle (social feed reading, analysis, decision generation, posting).
- `_desire_agent(...)`: information-demand generation + batch semantic news retrieval.
- `_intention_agent(...)`: generates social output (`post`, `type`, `belief`) from conversation context.
- `_polish_decision(...)`: sanitizes model outputs into executable and constrained trading decisions.
- `_process_decision_result(...)`: converts target-position decisions into quantity/price executable orders.
- `_get_price_limits(...)`: computes safety bounds around prior close (currently ±10% for A-share constraints).

**Why reuse:** this is the strongest reusable layer for injecting follower/leader prompts, copy-trading constraints, and anti-bias post-processing before execution.

## 3) Social graph and leader extraction

### `util/UserDB.py`
- `build_graph_new(...)`: builds time-decayed weighted user graph from trading behavior similarity.
- `get_top_n_users_by_degree(...)`: selects high-centrality users (natural "leaders").
- `update_graph(...)`: updates graph structure as behavior evolves over time.
- `get_all_user_ids(...)` / `get_user_profile(...)`: user universe and profile hydration.

**Why reuse:** this module is the direct foundation for identifying leaders and assigning follower neighborhoods.

## 4) Forum/reaction mechanics (social layer)

### `util/ForumDB.py`
- `init_db_forum(...)`: social tables (`posts`, `reactions`, `post_references`).
- `recommend_post_graph(...)`: graph-neighbor-constrained content ranking with time decay.
- `execute_forum_actions(...)`: executes actions (`repost`, `like`, `unlike`) asynchronously.
- `update_posts_score_by_date_range(...)`: computes engagement-based score propagation.
- `get_all_users_posts_db(...)`: belief extraction source by user/time.

**Why reuse:** gives immediate follower-feed mechanics and measurable social influence channels.

## 5) Execution and market microstructure

### `trader/matching_engine.py`
- `Order`: normalized order object.
- `calculate_closing_price(...)`: matching and clearing logic with price-time priority and limit constraints.
- `test_matching_system(...)`: daily E2E execution entrypoint.
- `update_profiles_table_holiday(...)`: profile carry-forward in no-trade periods.

**Why reuse:** can be ported to crypto by replacing session assumptions and constraints (e.g., remove market-close auction, add continuous matching/fees/slippage).

## 6) Belief initialization and behavioral priors

### `trader/init_belief.py`
- `init_belief(...)`: pipeline for generating initial belief text at scale.
- `process_dataframe(...)` / `process_chunk(...)`: concurrent belief generation.
- `retry_belief_conversion(...)`: robust generation fallback.

**Why reuse:** ideal for cold-starting follower/leader sentiment priors and later integrating de-biasing prompts.

## 7) Information retrieval layer

### `util/InformationDB.py`
- `load_database(...)`: restores FAISS + metadata store.
- `search_news(...)` / `search_news_batch(...)`: semantic retrieval by time range.

**Why reuse:** can index crypto-native sources (on-chain events, exchange notices, governance posts, X/Reddit) without changing agent interface.

## 8) Prompt and behavior policy layer

### `trader/prompts.py`
- `TradingPrompt.get_system_prompt_new(...)`: persona + behavior policy envelope.
- `TradingPrompt.get_user_first_prompt(...)`: injects account state and prior belief into context.

**Why reuse:** easiest place to add explicit "leader/follower role cards" and bias mitigation instructions.

## 9) Agent connector abstraction

### `Agent.py`
- `BaseAgent.get_response(...)`: unified model call interface.
- retry-enabled `__call_api(...)`: resilience for production-scale multi-agent runs.

**Why reuse:** swap model provider without changing higher-level social trading logic.

---

## Practical migration plan for follower/leader + bias mitigation

1. **Fork roles in prompt layer**
   - Add `role_type in {leader, follower}` and role-conditioned prompt templates in `trader/prompts.py`.
2. **Leader detection from graph + performance**
   - Start from `get_top_n_users_by_degree(...)`, then blend with Sharpe/return stability from profiles.
3. **Follower execution policy in `_polish_decision(...)`**
   - Blend self decision with leader basket under cap constraints:
     `final = alpha * self + (1-alpha) * leader_signal`.
4. **Bias mitigation insertion points**
   - Pre-decision: rebalance input set (counter-opinion/news diversity).
   - In-decision: constrain concentration and overreaction.
   - Post-decision: rejection/filter for herding and recency spikes.
5. **Crypto market adaptation**
   - Replace trading-day logic in `simulation.py` and matching assumptions in `trader/matching_engine.py` for continuous-time operation.
6. **Feedback loop metrics**
   - Add per-user metrics: herding score, conviction drift, source diversity, leader dependency ratio.

