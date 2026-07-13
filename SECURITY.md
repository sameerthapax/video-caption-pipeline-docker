# Security Policy

## Secrets

Never commit provider keys or proxy tokens. Use `.env` for the Python runtime and `caption-proxy/.dev.vars` for local Worker development; both are ignored by Git. Public templates are provided as `.env.example` files with blank values.

For a deployed Cloudflare Worker, configure secrets with Wrangler:

```bash
cd caption-proxy
npx wrangler secret put OPENROUTER_API_KEY
npx wrangler secret put FIREWORKS_API_KEY
npx wrangler secret put PROXY_TOKEN
```

Treat any credential that has appeared in Git history as compromised. Revoke or rotate it before making the repository public, even if it has since been deleted from the current files.

## Reporting

Please report security issues privately to the repository owner instead of opening a public issue.
