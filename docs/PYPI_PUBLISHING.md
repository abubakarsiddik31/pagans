# PyPI Publishing Guide

This project is ready to publish as a package named `pagans`.

## 1) One-time setup

1. Create a PyPI account: <https://pypi.org/account/register/>
2. Create an API token in PyPI account settings.
3. Add the token as a GitHub repository secret:
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-...`

## 2) Pre-release checks

Run locally:

```bash
uv sync --dev
uv run pytest -q
uv run python -m build
uv run twine check dist/*
```

## 3) Versioning

This project uses `setuptools_scm` (git-tag based versioning).

Create a release tag from `main`:

```bash
git checkout main
git pull

git tag v0.2.0
git push origin v0.2.0
```

Tag format should be `vX.Y.Z`.

## 4) Publish flow

Publishing is automated by GitHub Actions workflow `.github/workflows/publish.yml`.

- Trigger: pushing a tag matching `v*` or manual workflow dispatch.
- The workflow builds `sdist` and `wheel`.
- It then uploads to PyPI using `PYPI_API_TOKEN`.

## 5) Verify release

- Check package page: <https://pypi.org/project/pagans/>
- Test install in clean env:

```bash
python -m venv /tmp/pagans-test
source /tmp/pagans-test/bin/activate
pip install pagans
python -c "import pagans; print('ok')"
```
