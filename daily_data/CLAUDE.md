# Core Principles

These are non-negotiable. Every decision, analysis, and piece of code must adhere to these without exception. No principle may be relaxed to save time, tokens, or compute.

## 1. Never fabricate data
Never generate synthetic, placeholder, or simulated data to stand in for real market data. If real data is unavailable, surface the failure clearly rather than silently filling in made-up values.

## 2. Be thorough — never cut corners
Always prioritize completeness and precision. Never skip steps, truncate datasets, use approximations, or reduce granularity to save compute, tokens, API calls, or time.

## 3. Never silently accept missing data when it can be obtained
If data is missing, don't silently skip it. Quantify the gap, surface it immediately, and offer to fetch it. Never present analysis with silent holes.

## 4. Surface problems, don't hide them
If something looks wrong, flag it immediately. Never silently "fix" discrepancies or present results that paper over known issues. Bad news early is better than bad news late.

## 5. Forward-walking only
Never use hindsight or peak detection for entry/exit timing. All signals and features must use only data available at the moment of decision. If you catch yourself using future information, stop and redesign.

## 6. All backtest logic must match live strategy logic exactly
No shortcuts, simplifications, or idealizations in backtesting that wouldn't exist in the live execution path. If you can't do it live, you can't do it in the backtest.

## 7. Never editorialize without data
Do not interpret raw statistics as trading conclusions. Present numbers, let the data speak. "This looks promising" is not allowed without a t-stat.

## 8. Separate concerns — don't redo work that isn't broken
Before executing a plan, identify which parts are actually affected by the change and which are not. Surgically fix what's broken and preserve what's correct. Wasted compute is wasted time.

---

# Data Integrity Rules

## 9. Never assume intra-bar sequence from OHLC data
When simulating fills, targets, and stops using OHLC bars (any timeframe), you CANNOT determine the order of events within that bar. If a bar touches both your target and stop, you do NOT know which happened first. Never credit a fill AND a target/stop on the same bar without tick-level verification. When in doubt, assume worst case.

## 10. Never cross-contaminate symbols
All rankings, quintile cuts, and signal thresholds must be computed WITHIN each symbol. Never pool signals across symbols — each has its own volatility, microstructure, and edge profile.

## 11. Never use theoretical pricing models as a substitute for real data
Don't use Black-Scholes, binomial models, or Greeks-based estimation to generate prices/P&L when real market data is available. Models are supplementary only.

## 12. Always verify time-of-day mappings against actual clock time
When filtering by bar index or any sequential numbering, ALWAYS verify what clock time those indices correspond to. Pre-market and extended-hours data shifts numbering. Print actual timestamps and confirm before trusting.

## 13. Use bps-based thresholds, not fixed-cent thresholds
Fixed-cent stops/targets create price-level bias across time (SPY at $300 vs $600 makes 8¢ mean different things). Always use basis-point thresholds for any cross-period comparison or strategy parameter.

---

# Backtesting & Validation Rules

## 14. Null model testing is mandatory
Always test whether random entries produce similar results to your signal. If your signal doesn't meaningfully beat random, the edge is structural (e.g., limit order mechanics), not signal-driven. Never skip this step.

## 15. Walk-forward parameter instability = overfit
If the optimal parameters change in every walk-forward split, the in-sample optimization is fitting noise. Stable parameters across windows are a prerequisite for trusting a strategy.

## 16. Fill-minute ambiguity must be resolved
If a large fraction of profits come from same-bar fills where sequence can't be verified, the backtest is unreliable. Use tick data to verify or assume worst case. Never report results that depend on unverifiable fill ordering.

## 17. Execution realism kills thin edges
Always stress-test with realistic slippage, latency, and adverse selection. Edges that exist statistically but die at 1 second of delay or $0.03/share slippage are not tradeable edges — they are backtest artifacts. Model the execution, not just the signal.

## 18. Market makers price microstructure correctly
Option bid-ask spreads absorb underlying microstructure edges. If the underlying effect is 5-40 bps and option spreads cost 10-30 bps, the edge is dead on arrival. Check this BEFORE building an options strategy on top of a statistical finding.

## 19. Reactive entry on fast mean-reversion has zero edge
The reversion happens before you can react. Pre-positioned resting orders are the only viable approach for capturing intra-second microstructure. Don't waste time building reactive signal → order pipelines for sub-second phenomena.

---

# Workflow Rules

## 20. Always save outputs to files
Never just print results to terminal. Every analysis output, summary, chart, and script must be saved to disk.

## 21. Save scripts to disk
Scripts must survive session restarts. Never rely on in-memory-only code.

## 22. Think before structural changes
Before executing any structural change (moving files, renaming folders, changing configs, modifying scripts used by running processes), stop and think through what could break. Check for running background processes first.

## 23. Update the work log
At the end of each major task, append a summary to the Work Log section below.

---

# Data Source
- **API**: Polygon.io
- **API Key**: `cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF`

---

# Project Goal

Find tradeable statistical edge in daily equity data across a broad mid/large-cap universe, using proper quant methodology.

---

# Current Strategy (under investigation)

**Signal:** z_sma50_slope — z-scored 5-10 day change in 50-day SMA, within each symbol
**Trade:** Cross-sectional L/S quintile sort across 230 mid/large-cap names
**Long:** Bottom quintile (SMA50 slope declining fastest — trend deteriorating)
**Short:** Top quintile (SMA50 slope rising fastest — trend accelerating)
**Thesis:** Stocks whose medium-term trend is accelerating in either direction are overshooting; capture mean reversion when acceleration fades

## Clean Holdout Results (2023-2026, strategy frozen from 2020-2022 validation)

| Variant | Hold | Val SR (2020-22) | Holdout SR (2023-26) | Holdout Net bps | Years |
|---|---|---|---|---|---|
| **sma50_slope_10d, Q5, EW** | 1d | 4.99 | **7.18** | +63.6 | 4/4 |
| sma50_slope_5d, Q5, VE | 1d | 3.82 | **6.81** | +48.9 | 4/4 |
| sma50_slope_10d, Q5, EW | 5d | 4.78 | **7.74** | +311.4 | 4/4 |

Both legs profitable: long +31 bps, short +33 bps. Turnover 4%/day. Cost drag <1 bps.

## Open Questions / Red Flags

1. **Sharpe 7+ is implausibly high** for public price data. Missing: borrow costs, market impact, execution friction.
2. **No sector neutrality tested** — could be a concentrated sector bet (long beaten-down energy, short rising tech).
3. **No borrow cost modeled** on short leg. Hard-to-borrow names could add 1-3% annualized drag.
4. **Survivorship bias in universe** — 230 names selected as "in play" with hindsight.
5. **Holdout IMPROVED vs validate** — unusual, possibly regime-dependent (2023-26 steady uptrend).
6. **Factor regression shows R²=0.9% vs FF5+Mom** — either genuinely orthogonal or our regression is wrong.
7. **Need paper trading** with live data before any capital allocation.

---

# Data

## Universe
- 230 mid/large-cap names across 17 sectors + 27 ETFs
- Downloaded from Polygon.io, 2015-01-01 to 2026-04-03
- All split-adjusted daily OHLCV + VWAP + transaction count
- Files: `data/{SYM}_daily.csv` (raw), `data/{SYM}_enriched.csv` (64 cols with indicators)

## Indicators (64 columns per file)
BB (upper/lower/mid/%B/bandwidth), RSI(14), SMA(8/12/20/50/100/200), EMA(8/12/20/50),
ATR(14), volume metrics (SMA20/50, rvol), MACD, ADX(14), Stochastic, ROC(10/20), OBV,
Keltner Channels, MA distance %, returns, realized vol, range/gap bps, candle structure,
MA slope, squeeze, consecutive days, proximity to 20d/52w highs/lows

## Fama-French Factors
- `data/ff5_daily.csv`: Mkt-RF, SMB, HML, RMW, CMA, RF (2015-2026)
- `data/mom_daily.csv`: Momentum factor (2015-2026)

---

# Scripts

```
scripts/
├── level1_scan.py              ← Broad 82-condition × 4-hold scan across 230 symbols
├── level2_deep_dive.py         ← Walk-forward, null model, slippage on 28 candidates
├── data_driven_scan.py         ← IC ranking, quintile spreads, interactions (no priors)
├── level3_proper.py            ← Excess returns, z-scored features, train/test split
├── level4_symmetry.py          ← L/S leg decomposition, factor overlap, turnover
├── level5_production.py        ← Dollar P&L, proxy factor regression, capacity
├── level6_fixes.py             ← Real FF5+Mom, lagged rolling mean, vol-equalization
└── level7_clean_holdout.py     ← Proper 3-way split: train/validate/holdout
```

# Analysis Results

```
analysis_results/
├── level1_scan_report.txt              ← 169/324 BH-significant, all long-side MR
├── level1_pooled_results.csv           ← All pooled test results
├── level1_per_symbol_results.csv       ← 62K per-symbol tests
├── level2_deep_dive_report.txt         ← 18 pass, 10 killed. All are "buy the dip"
├── level2_summary.csv
├── data_driven_scan_report.txt         ← IC scan: SMA slope is top non-price feature
├── data_driven_ic_all.csv
├── data_driven_quintiles.csv
├── data_driven_tails_*.csv
├── data_driven_interactions_*.csv
├── level3_proper_report.txt            ← Everything survives train→test on excess returns
├── level3_final_results.csv
├── level4_symmetry_report.txt          ← Symmetric: short leg SR > long leg SR
├── level4_symmetry.csv
├── level5_production_report.txt        ← $17M/yr at $10M/leg, costs irrelevant
├── level6_fixes_report.txt             ← FF5 R²=0.9%, vol ratio 1.04x, SR→4.64
└── level7_holdout_report.txt           ← Clean holdout: SR 7.18, 4/4 years positive
```

---

# Key Findings (methodology, not strategy-specific)

1. **Most textbook TA signals are just the equity risk premium in disguise.** RSI<30, BB below lower, consecutive down days — all just "buy cheap stocks that go up because stocks go up."
2. **Must subtract each symbol's own drift before pooling.** Otherwise every long-side signal looks good.
3. **Must z-score features within each symbol before cross-sectional comparison.** A 10% drawdown means different things for TSLA vs JNJ.
4. **Golden crosses, MACD crosses, squeeze breakouts: all dead.** No statistical significance after multiple testing correction.
5. **Short-side mean reversion doesn't work on raw returns** because long-term equity drift kills it. Only works on excess returns.
6. **SMA50 slope is the Goldilocks feature** — fast MAs (8,12) are noisy, slow MAs (200) are too sluggish.
7. **Optimization on validate period doesn't help much.** Pre-specified primary (SR 6.81) nearly matches optimized (SR 7.18).
8. **Vol-equalization reduces Sharpe by ~15%** but doesn't change direction of results. Vol asymmetry between legs was only 1.04x (not the 2-3x the lead quant feared).

---

# Work Log

### 2026-04-03
- Downloaded daily bars for 230 symbols (2015-2026) from Polygon.io → `data/{SYM}_daily.csv`
- Built 64-column indicator set → `data/{SYM}_enriched.csv`
- **Level 1**: Broad scan of 82 conditions × 4 hold periods. 169/324 BH-significant. All survivors are long-side mean reversion ("buy the dip").
- **Level 2**: Deep dive on 28 candidates with walk-forward, null model, slippage breakeven. 18 pass. Every surviving strategy is the same trade: buy weakness, hold 3-5 days.
- **Data-driven scan**: Removed priors, ran IC ranking on excess returns. Top features are all price level proxies. Non-obvious finding: `sma50_slope_5d` (how fast the MA is changing, not where price is).
- **Level 3**: Proper excess returns (subtract rolling mean), z-scored features, train/test split. Everything survives. 100% consistency across 230 symbols.
- **Level 4**: Symmetry test — both legs profitable (short side SR > long side). MA50 is optimal MA period. Signal is 1.78x better Sharpe than vanilla reversal with 6x less turnover. Correlation with 5d reversal: 0.47.
- **Level 5**: Dollar P&L with costs. $17M/yr at $10M/leg. Cost drag 1% of gross. Capacity: $20M/leg at 1% ADV of smallest name. Proxy factor R²=12.8%.
- **Level 6**: Three fixes — real FF5+Mom factors (R² dropped to 0.9%), lagged rolling mean (minimal impact), vol-equalization (SR from 5.45 to 4.69). Net SR: 4.64.
- **Level 7**: Clean 3-way split. Train 2015-19, validate 2020-22, holdout 2023-26. Holdout IMPROVED vs validate (SR 7.18 vs 4.99). 4/4 years positive. All 10 robustness variants pass. Implausibly high Sharpe remains a red flag.
- Downloaded FF5 + momentum daily factors from Ken French's website
