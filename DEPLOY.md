# Deployment Guide

## Quick Deploy to Railway (Recommended)

### Option 1: Railway Dashboard (Easiest)

1. Go to [railway.app](https://railway.app) and log in
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Connect your GitHub account and select `q5-momentum-dashboard`
4. Railway auto-detects Next.js and deploys
5. Go to **Variables** tab and add:
   ```
   POLYGON_API_KEY = cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF
   ```
6. Railway provides a live URL like `q5-momentum-dashboard.up.railway.app`

### Option 2: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to project (from repo root)
cd q5-momentum-dashboard
railway link

# Set environment variable
railway variables set POLYGON_API_KEY=cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF

# Deploy
railway up
```

### Option 3: Auto-deploy via GitHub Actions

1. Get your Railway token from: https://railway.app/account/tokens
2. Go to your GitHub repo → Settings → Secrets → New repository secret
3. Add secret: `RAILWAY_TOKEN` = your railway token
4. Any push to `main` will auto-deploy

---

## Deploy to Vercel (Alternative)

Vercel has native Next.js support:

```bash
npm install -g vercel
cd q5-momentum-dashboard
vercel
# Follow prompts, add POLYGON_API_KEY when asked for env vars
```

Or via Dashboard:
1. Go to [vercel.com](https://vercel.com) and import from GitHub
2. Add environment variable: `POLYGON_API_KEY=cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF`

---

## Environment Variables

| Variable | Value | Required |
|----------|-------|----------|
| `POLYGON_API_KEY` | `cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF` | Yes |
| `PORT` | `3000` (auto-set by Railway) | No |

## Build Commands

```bash
npm run build   # Build production bundle
npm run start   # Start production server
npm run dev     # Start development server
```

## GitHub Repository

https://github.com/ztariff/q5-momentum-dashboard
