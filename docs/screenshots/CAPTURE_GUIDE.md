# MarketPulse portfolio screenshots

The repository currently has no genuine dashboard screenshots. Capture these from the locally running Streamlit application and save the resulting PNG files in this directory. Do not use mock data or edit values/statuses to imply a feature is working when it is not.

## Recommended captures

| Filename | Page and suggested view | Include |
|---|---|---|
| `overview.png` | Overview, page at top | MARKETPULSE header, all six watchlist instruments if practical, provider/freshness labels, and recent alerts or its honest empty state |
| `volatility.png` | Volatility, select one instrument and `1h` | Price/volatility charts, key metric cards, and any timestamp-trust or unavailable indicators that apply |
| `liquidity-order-flow.png` | Liquidity & Order Flow, select BTCUSDT or ETHUSDT | Book synchronization/freshness, bid/ask depth, OBI, and trade-flow charts; keep the displayed-liquidity caveat visible if present |
| `market-regime.png` | Market Regime, summary and instrument evidence | Current classifications, feature values, evidence status and timestamp trust |
| `anomaly-detection.png` | Anomaly Detection, instrument results | Warm-up/model status, prior-observation progress, decision/score when available, and current feature count |
| `alerts-center.png` | Alerts Center | Real persisted alerts and filters, or the genuine empty state; do not create alerts for the screenshot |

## Capture checklist

1. Start the app from the project root with `streamlit run app.py` and wait for the selected page to render.
2. Use a consistent desktop viewport and browser zoom across captures. Prefer a viewport wide enough to show the page structure without clipping labels.
3. Capture the actual application window. Do not paste in synthetic prices, scores, alert rows or market values.
4. Check each image for account numbers, terminal login details, personal paths, credentials, browser notifications or other private information. Redact private UI chrome before committing; never redact or alter a market metric in a way that changes its meaning.
5. Use the exact lowercase filenames above so the README gallery can link to them consistently.

When all images exist, add them to the README with relative Markdown paths such as `docs/screenshots/overview.png`. Until then, the README explicitly describes the gallery as pending real captures.
