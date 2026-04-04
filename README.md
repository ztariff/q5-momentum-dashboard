# Q5 Momentum Strategy Dashboard

A live trading strategy dashboard for the Q5 Momentum strategy. Built with Next.js, Tailwind CSS, and Polygon.io for live market data.

## Strategy Overview

The Q5 Momentum strategy:
1. Scans 230 stocks daily for SMA50 slope acceleration
2. Enters when a stock first enters the top quintile (Q5) of z-scored slope values
3. Holds for 20 days (stock) or 13 days (options)
4. Uses ATR-based stop: `signal_day_low - 3 × ATR(14)`
5. Tiered sizing based on candle body quality (A/B/C tiers)

**Backtest results (2020–2026):**
- Total P&L: +$88.2M
- Win Rate: 55.3%
- Profit Factor: 2.03
- Sharpe: 0.61
- Max Drawdown: -$6.1M

## Dashboard Sections

1. **Current Positions** — Live open positions with unrealized P&L and stop distances
2. **Risk Alerts** — Positions near stop, near exit, or in deep loss
3. **Signal Scanner** — Daily scan for new Q5 entries across all 230 symbols
4. **Performance** — Equity curve, monthly/yearly P&L charts, and strategy metrics

## Setup

### Local Development

```bash
npm install
cp .env.example .env.local
# Edit .env.local with your Polygon.io API key
npm run dev
```

### Deploy to Railway

1. Push this repo to GitHub
2. Go to [Railway.app](https://railway.app) and create a new project
3. Connect your GitHub repo
4. Set environment variable: `POLYGON_API_KEY=your_key_here`
5. Railway auto-detects Next.js and deploys

## Environment Variables

| Variable | Description |
|----------|-------------|
| `POLYGON_API_KEY` | Polygon.io API key for market data |

## Tech Stack

- **Framework:** Next.js 15 (App Router)
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **Icons:** Lucide React
- **Data:** Polygon.io REST API
- **Deployment:** Railway
