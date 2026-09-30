# Contributing

Contributions that improve reproducibility, portability, documentation, or
correctness are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
ruff format --check .
ruff check .
pytest
```

## Change guidelines

- Keep reusable implementation under `src/sparmoe_vl/`; experiment folders
  should remain thin entry points and immutable protocol configuration.
- Do not commit datasets, pretrained weights, generated metrics, plots,
  caches, or machine-specific absolute paths.
- Preserve the registered experiment seeds, ordered sample counts, and SHA-256 data
  identities unless the change explicitly defines a new protocol.
- Add focused CPU tests for behavior changes and update the relevant public
  documentation.
- Keep optional LLaVA and analysis dependencies behind their package extras.

When reporting a bug, include the command, operating system, Python and
PyTorch versions, accelerator type, relevant configuration, and complete
traceback. Do not attach private data or model weights to a public issue.
