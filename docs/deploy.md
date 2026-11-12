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

This stays up without your Mac. MLX does not run on Linux, so the judge is Ollama. A 4-bit 9B model wants about 16 GB of RAM. Without a GPU it is much slower than MLX on Apple silicon, and each check judges several passages. If you change the model, run `make bench` again, because the v0.2.0 numbers describe Qwen3.5 9B on MLX.

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

## Vercel and other serverless hosts

A serverless function cannot run the local model: there is no GPU, and the weights are far larger than a function bundle. Two setups can still work:

- **No model.** Set `VERIFAI_MODEL_BACKEND=none`. The citation checker works fully, and it scored 16 of 16 in the v0.2.0 benchmark. Claims list their sources and stay unverified.
- **The model elsewhere.** Point `VERIFAI_OLLAMA_BASE_URL` at an Ollama server that the function can reach. That server is Option B without Caddy in front of VerifAI, and it needs its own protection, because Ollama has no authentication.

On serverless, the in-process limits above do not hold: each instance counts on its own, and instances come and go. Use the host's own rate-limit rules in front of the app. The repository does not include a Vercel configuration yet, and this setup has not been tested.

## The browser extension

The extension's default server is `http://127.0.0.1:8741`, the app on the same computer. To use a public server, open the extension's options, paste the server's https address, and save. The browser asks once for permission to reach that address. Nothing else on the web is granted: `optional_host_permissions` lets the extension ask, and it asks only for the server you saved.

When a public server refuses a check, the panel shows the server's reason, for example its rate limit.

Before you publish the extension for one public server, make that address the default in `extension/src/config.js` and add it to `host_permissions`, so a new user does not have to paste it. A quick-tunnel address changes every time the tunnel restarts, so wait for a named tunnel or a domain.
