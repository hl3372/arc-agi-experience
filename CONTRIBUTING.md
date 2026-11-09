# Contributing

Bug reports and fixes are welcome: please open an issue or a pull request.

## Repository layout

One module per top-level folder (`generator/`, `research-envs/`, …). Each module:

- **is self-contained:** its own `README.md` (what it is, install, quickstart, configuration), `pyproject.toml` or
  `requirements.txt`, and `tests/`;
- **runs from its own folder:** no absolute paths or machine-specific locations. Data and outputs go to configurable,
  git-ignored folders;
- **keeps keys out of the code:** keys and tokens are read from environment variables, never committed;
- **has an offline test suite** that needs no network and no model calls.

When you add a module, add a row for it to the table in the top-level `README.md`.

## Before committing

- Run the module's tests.
- Check that nothing private ships:
  - no API keys or tokens;
  - no personal paths, user names or e-mail addresses;
  - no ARC-AGI-2 evaluation tasks, IDs or annotations;
  - no large data files. Publish data separately, for example on Hugging Face.
