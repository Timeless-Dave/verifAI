# Putting VerifAI on a public link

VerifAI is built to run on your own machine. A public link means strangers' checks run on your hardware, and every check also calls Crossref, OpenAlex, PubMed, Wikipedia, and the other sources with your `VERIFAI_MAILTO` address. Turn the limits on before you share the link.

## Limits

| Variable | Default | For a public link |
| --- | --- | --- |
| `VERIFAI_RATE_LIMIT_PER_MINUTE` | `0` (off) | `5` |
| `VERIFAI_RATE_LIMIT_PER_HOUR` | `0` (off) | `40` |
| `VERIFAI_MAX_CONCURRENT_CHECKS` | `4` | `2` with a local model |
| `VERIFAI_CHECK_QUEUE_SECONDS` | `20` | `20` |
| `VERIFAI_FORWARDED_ALLOW_IPS` | `127.0.0.1` | the proxy's address, if it is not on the same machine |

The limits count checks per client address. A refused API call gets HTTP 429 (rate limit) or 503 (busy) with a `Retry-After` header. The web form shows the same message in place of results. The home page, static files, and `/health` are never limited.

Counts live in the VerifAI process. That is right for one `verifai` process on one machine, which is how `make run` starts it. If you run several processes or serverless instances, each keeps its own counts, so put a shared limit in front of them.

When VerifAI listens on an address other than localhost with no rate limit, it logs a warning at startup.

### Client addresses behind a proxy

VerifAI sees the proxy as the client unless it trusts the proxy's `X-Forwarded-For` header. It trusts `127.0.0.1` by default, so Caddy, nginx, or cloudflared on the same machine work with no change. If the proxy runs somewhere else, for example on a Docker bridge, set `VERIFAI_FORWARDED_ALLOW_IPS` to its address. Do not set it to `*` unless only the proxy can reach the port: anyone else could then pick their own address and skip the limit.

Behind a proxy on the same machine, keep `VERIFAI_HOST=127.0.0.1`. The proxy is the only thing that needs to reach VerifAI.

## Option A: a tunnel from your Mac

This keeps MLX and the published benchmark numbers. The link works only while the Mac is awake and `make run` is running.

Add to `.env`:

```bash
VERIFAI_MODEL_BACKEND=mlx
VERIFAI_RATE_LIMIT_PER_MINUTE=5
VERIFAI_RATE_LIMIT_PER_HOUR=40
VERIFAI_MAX_CONCURRENT_CHECKS=2
```

Then, in two terminals:

```bash
make run
```

```bash
brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8741
```

The second command prints a random `https://….trycloudflare.com` address. No Cloudflare account is needed, and the address changes every time you restart it. For a fixed address on your own domain, create a named tunnel with a free Cloudflare account:

```bash
cloudflared tunnel login
cloudflared tunnel create verifai
cloudflared tunnel route dns verifai verify.example.com
cloudflared tunnel run --url http://127.0.0.1:8741 verifai
```

Check that the limit sees separate visitors: open the link on your phone, run a few checks, and look for the visitors' addresses in the `make run` log. If every line shows `127.0.0.1`, the tunnel is not passing `X-Forwarded-For`, and every visitor shares one budget.

macOS sleeps by default. `caffeinate -s make run` keeps the Mac awake while the server runs on power.

## Option B: a Linux server

This stays up without your Mac. MLX does not run on Linux, so the judge is Ollama. A 4-bit 9B model wants about 16 GB of RAM. Without a GPU it is much slower than MLX on Apple silicon, and each check judges several passages. If you change the model, run `make bench` again, because the published numbers describe Qwen3.5 9B on MLX.

On an Ubuntu or Debian server with a domain pointed at it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:9b
sudo apt install caddy
git clone https://github.com/Timeless-Dave/verifAI.git && cd verifAI
cp .env.example .env
```

In `.env`, set `VERIFAI_MAILTO`, `VERIFAI_MODEL_BACKEND=ollama`, and the limits from the table above. Leave `VERIFAI_HOST=127.0.0.1`.

`/etc/caddy/Caddyfile` (Caddy gets the HTTPS certificate itself):

```
verify.example.com {
	reverse_proxy 127.0.0.1:8741
}
```

`/etc/systemd/system/verifai.service`, with `User` and the paths changed to yours:

```ini
[Unit]
Description=VerifAI
After=network-online.target ollama.service

[Service]
User=verifai
WorkingDirectory=/home/verifai/verifAI
ExecStart=/home/verifai/.local/bin/uv run --no-sync verifai
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
make install EXTRAS=
sudo systemctl daemon-reload
sudo systemctl enable --now verifai
sudo systemctl reload caddy
```

## Vercel

The public demo, https://useverifai.vercel.app, runs this way. `api/index.py` is the entry point and `vercel.json` sends every path to it. Vercel installs the project from `pyproject.toml` and `uv.lock`, without the model extras.

A serverless function cannot run a model itself: there is no GPU, and the weights are far larger than a function. The demo uses `VERIFAI_MODEL_BACKEND=hosted`: an open-weights model on Cloudflare Workers AI, reached through its OpenAI-compatible API. Set `VERIFAI_HOSTED_BASE_URL=https://api.cloudflare.com/client/v4/accounts/<account id>/ai/v1`, `VERIFAI_HOSTED_API_KEY` to a Workers AI token, and optionally `VERIFAI_HOSTED_MODEL` (default `@cf/qwen/qwen3-30b-a3b-fp8`). Any OpenAI-compatible provider works the same way.

Environment variables for the project (`vercel env add NAME production`):

| Variable | Value |
| --- | --- |
| `VERIFAI_MAILTO` | an address you read |
| `VERIFAI_MODEL_BACKEND` | `hosted` |
| `VERIFAI_NLI_ENABLED` | `false` |
| `VERIFAI_RATE_LIMIT_PER_MINUTE` | `5` |
| `VERIFAI_RATE_LIMIT_PER_HOUR` | `40` |
| `VERIFAI_CLIENT_IP_HEADER` | `x-real-ip` |

Vercel sets `x-real-ip` to the visitor's address, and a visitor cannot override it. `VERIFAI_CLIENT_IP_HEADER` tells the rate limiter to trust it. Do not set it on a server where visitors reach VerifAI directly, because there anyone can send that header.

Each function instance keeps its own counts, and instances come and go, so the in-app limit is softer on Vercel than on one long-running server. For a hard limit, add a rate-limit rule in the project's Vercel Firewall as well.

```bash
vercel link --project useverifai
vercel deploy --prod
```

`.vercelignore` keeps `.env` and other local files out of the upload.

## Lend your Mac's model to the public site

Vercel cannot run a model, but it can borrow one. A VerifAI with a model answers `POST /api/v1/judge` when `VERIFAI_JUDGE_TOKEN` is set, and only to callers that send that token. The public site uses it through `VERIFAI_MODEL_BACKEND=remote`. When the Mac is asleep or off, the site checks whether it is reachable at most every few seconds, falls back to citations only, and says claims were not judged.

On the Mac:

```bash
# .env: VERIFAI_JUDGE_TOKEN=<a long random secret>
scripts/macos/verifai-agent.sh install   # starts VerifAI at login, restarts it if it stops
```

The Mac also needs a public HTTPS address that does not change, for example Tailscale Funnel or a Cloudflare named tunnel on your own domain. Then, on Vercel:

| Variable | Value |
| --- | --- |
| `VERIFAI_MODEL_BACKEND` | `remote` |
| `VERIFAI_REMOTE_JUDGE_URL` | the Mac's public address |
| `VERIFAI_JUDGE_TOKEN` | the same secret |
| `VERIFAI_MAX_CLAIMS` | `3`, so a batch fits Vercel's 60-second limit |

Judging is one passage at a time on the Mac, in order, because MLX cannot run two at once.

## The browser extension

The extension's default server is https://useverifai.vercel.app, which `host_permissions` already allows. To have claims judged by a local model, set the server to `http://127.0.0.1:8741` in the extension's options. To use another public server, paste its https address there and save. The browser asks once for permission to reach that address. Nothing else on the web is granted: `optional_host_permissions` lets the extension ask, and it asks only for the server you saved.

When a public server refuses a check, the panel shows the server's reason, for example its rate limit.

A fork that runs its own public server changes both places: `DEFAULT_SERVER` in `extension/src/config.js` and `host_permissions` in `extension/manifest.json`. Use an address that does not change. A quick-tunnel address changes every time the tunnel restarts.
