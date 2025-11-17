# Pullus Competitor Analysis

Automated market intelligence system for monitoring competitor pricing in the Nigerian poultry market.

## Overview

This system automatically analyzes Pullus pricing against competitors across multiple locations (Abuja, Kaduna, Kano) for various chicken products. It generates visual dashboards, executive scorecards, and email alerts to support pricing decisions.

## Features

- **Same-day competitive analysis** - Compares Pullus prices against competitors for 14 product categories
- **Automated dashboards** - Creates management sheets in Google Sheets with color-coded insights
- **Smart change detection** - Only processes updates when source data actually changes (saves ~80% compute)
- **Daily email alerts** - Sends notifications when new market data is available
- **Confidence scoring** - Evaluates data reliability based on competitor sample size

## Tech Stack

- Python 3.11
- Google Sheets API (gspread)
- pandas/numpy for data processing
- GitHub Actions for automation
- Gmail SMTP for alerts

## Key Components

- `pullus_competitor_analyzer.py` - Core analysis engine
- `check_changes.py` - Content-based change detection system
- `email_alerts.py` - Email notification module
- `config.yaml` - Configuration (thresholds, products, locations)

## Status Categories

- **🟢 BEST PRICE** - Pullus has the cheapest price
- **🟡 COMPETITIVE** - Within ±₦200 of cheapest competitor
- **🔴 ABOVE MARKET** - Significantly more expensive than competitors
- **⚪ NO DATA** - No competitor data available for comparison

## Automation

- **Continuous updates**: Runs every 30 minutes via GitHub Actions
- **Daily alerts**: Sends email at 8 AM WAT with latest insights
- **Change detection**: Prevents unnecessary runs when data hasn't changed

## Setup

Requires environment variables:
- `GOOGLE_CREDENTIALS_PATH` - Google service account credentials
- `GOOGLE_SHEET_ID` - Target spreadsheet ID

## Documentation

See `SMART_WORKFLOW_GUIDE.md` for details on the intelligent change detection system.
