# Security Policy

## Reporting a vulnerability

Please report security issues **privately** using GitHub's private vulnerability reporting —
the **Report a vulnerability** button under this repository's **Security** tab. Do not open a
public issue for a security report.

We aim to acknowledge a report within a few business days and will coordinate a fix and a
disclosure timeline with you.

## Scope

FieldLoop's OSS core runs locally with no network services by default, so the most
security-relevant areas are:

- **Tenant isolation** in the storage/serialization layer — a query or row must never cross
  a tenant boundary.
- **Fail-closed handling** in attribution and curation — an uncertain or retracted binding
  must never be silently promoted into trusted training data.

Optional live-database wiring (the `live-db` feature) is off by default; if you enable it,
the usual database-credential and network-exposure considerations apply.
