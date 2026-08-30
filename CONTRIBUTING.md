# Contributing to token-meter

Issues and pull requests are welcome. This file covers how to get the project
running locally and what a pull request is expected to carry.

## Local development

Requires Rust.

```bash
git clone https://github.com/asale-ai/token-meter
cd token-meter
cargo build
cargo test
```

## Before opening a pull request

- Run `cargo test` and make sure it passes.
- Keep the change focused: one concern per pull request is easier to review
  and easier to revert.
- If the behaviour changed, update the README in the same commit. Documentation
  that lags behind the code is worse than no documentation.

## Reporting a bug

Open an issue with the version you are running and the smallest set of steps
that reproduces the problem. A report without a reproduction usually costs a
few round trips before triage can even start.
