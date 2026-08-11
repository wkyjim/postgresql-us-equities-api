# Telegram Bot Webhook on Render

The Render API exposes a Telegram webhook through the existing FastAPI service.

## Render Environment Variables

Required:

- `neon_password`: existing Neon database password.
- `TG_token`: Telegram bot token from BotFather.
- `TG_chat_id`: allowed Telegram chat ID. Updates from other chats are ignored.
- `TG_webhook_secret`: random path secret used in `/telegram/webhook/{secret}`.
- `TG_RENDER_BASE_URL`: public Render base URL, for example `https://postgresql-us-equities-api.onrender.com`.

Recommended:

- `TG_webhook_secret_token`: random Telegram header secret used by Telegram's `secret_token` verification.
- `TG_REPORT_URL`: public raw/latest report URL. Use the GitHub Pages report markdown URL if available.

Optional aliases supported by the API:

- `TG_TOKEN`
- `TG_CHAT_ID`
- `TG_WEBHOOK_SECRET`
- `TG_WEBHOOK_SECRET_TOKEN`
- `RENDER_EXTERNAL_URL`
- `LATEST_REPORT_URL`

## Register Webhook

After Render deploys, call:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "https://postgresql-us-equities-api.onrender.com/telegram/webhook/register/<TG_webhook_secret>"
```

If `TG_webhook_secret_token` is configured, include:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "https://postgresql-us-equities-api.onrender.com/telegram/webhook/register/<TG_webhook_secret>" `
  -Headers @{ "X-Telegram-Bot-Api-Secret-Token" = "<TG_webhook_secret_token>" }
```

The registered Telegram webhook URL will be:

```text
https://postgresql-us-equities-api.onrender.com/telegram/webhook/<TG_webhook_secret>
```

## Bot Commands

- `/start`
- `/market`
- `/dashboard`
- `/report`
- `/signals`
- `/sectors`
- `/equity AAPL`
- `/macro`
- `/risk`
- `/status`

## Operational Notes

- `/market`, `/macro`, and `/equity` read live/fallback data from Neon through the existing API queries.
- `/dashboard`, `/report`, `/signals`, `/sectors`, and `/risk` use `TG_REPORT_URL` when configured.
- If `TG_REPORT_URL` is missing, `/dashboard` falls back to market tape and `/report` explains that the report URL is unavailable.
- Long Telegram messages are split into safe `sendMessage` chunks.
- Do not commit real Telegram tokens, chat IDs, or webhook secrets.
