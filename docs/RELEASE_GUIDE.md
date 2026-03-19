# GitHub Release Guide

This guide explains how to create releases for the PAGANS project using GitHub Actions.

## Prerequisites

1. **Set up PyPI API token**:
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Add it to your GitHub repository secrets as `PYPI_API_TOKEN`
   - Settings → Secrets and variables → Actions → New repository secret

2. **Repository permissions**:
   - Go to Settings → Actions → General → Workflow permissions
   - Enable "Read and write permissions"

## Creating a Release

### Option 1: Using GitHub UI (Recommended)

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Choose the tag version:
   - Format: `v1.0.0` (semantic versioning)
   - If the tag doesn't exist, GitHub will create it for you
4. Write release notes
5. Click "Publish release"
6. The publish workflow will automatically trigger and deploy to PyPI

### Option 2: Using Git Tags

```bash
# Create and push a version tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

The workflow will automatically trigger on the tag push.

### Option 3: Manual Workflow Trigger

1. Go to Actions tab in GitHub
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select the branch
5. Click "Run workflow"

## Workflow Triggers

The publish workflow is triggered by:

1. **GitHub Release** (when you create a release through the UI)
2. **Version tags** (when you push a tag starting with `v*`)
3. **Manual trigger** (through the Actions UI)

## CI/CD Pipeline

The complete pipeline is:

1. **Lint** → Runs code quality checks
2. **Test** → Runs tests on Python 3.12 and 3.13
3. **Build** → Creates distribution packages
4. **Publish** → Deploys to PyPI

Each workflow runs automatically based on triggers:

- `lint.yml` and `test.yml`: On push/PR to `main` or `dev`
- `build.yml`: After successful tests
- `publish.yml`: On releases/tags or manual trigger

## Version Management

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

Examples:
- `v1.0.0` → First stable release
- `v1.1.0` → New feature
- `v1.1.1` → Bug fix
- `v2.0.0` → Breaking changes

## Release Checklist

Before creating a release:

- [ ] All tests passing: `uv run pytest`
- [ ] Code is linted: `uv run ruff check` and `uv run ruff format`
- [ ] Types checked: `uv run mypy src/`
- [ ] Version updated in `pyproject.toml` (if needed)
- [ ] CHANGELOG.md updated
- [ ] Documentation updated (if needed)
- [ ] PyPI API token configured as repository secret

## Testing Before Release

To test the build locally:

```bash
# Build the package
uv run python -m build

# Check the package
uv run twine check dist/*

# The dist/ folder will contain:
# - .tar.gz (source distribution)
# - .whl (wheel distribution)
```

## Troubleshooting

### Publish workflow fails
- Check that `PYPI_API_TOKEN` is correctly set as a secret
- Verify the tag format matches `v*` pattern
- Ensure your PyPI account has the correct permissions

### Build fails
- Make sure all dependencies are listed in `pyproject.toml`
- Check that `build` and `twine` are in the dev dependencies
- Verify the package structure is correct

### Tests fail in CI but pass locally
- Ensure your local environment matches the CI (Python 3.12/3.13)
- Check for platform-specific issues
- Verify all test files are committed

## Automated Release Notes

When creating a release through GitHub UI:

1. GitHub automatically shows commit highlights
2. Add your custom release notes
3. Include:
   - New features
   - Bug fixes
   - Breaking changes
   - Migration instructions (if needed)
   - Links to documentation

## Post-Release Actions

After a successful release:

1. **Announce**: Share release notes with users
2. **Update docs**: Deploy updated documentation
3. **Monitor**: Watch for issues reported by users
4. **Plan next version**: Start working on the next release

## Additional Resources

- [Semantic Versioning](https://semver.org/)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)