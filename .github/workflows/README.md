# GitHub Actions CI/CD Pipelines

This directory contains the GitHub Actions workflows for the ScpTensor project.

## Workflows

### 1. CI - Continuous Integration (`ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs:**

| Job | Purpose | Matrix |
|-----|---------|--------|
| `quality` | Ruff lint, format check, mypy type check | Ubuntu |
| `test` | Pytest with coverage | Ubuntu/macOS/Windows × Python 3.12/3.13 |
| `build` | Build package verification | Ubuntu |
| `security` | Bandit security scanning | Ubuntu |

**Artifacts:**
- `coverage-report`: HTML coverage report
- `dist`: Built wheel/sdist packages
- `bandit-report`: Security scan results

---

### 2. CD - Continuous Deployment (`cd.yml`)

**Triggers:**
- Release published (automatic PyPI publish)
- Manual workflow dispatch (for testing)

**Jobs:**

| Job | Purpose | Condition |
|-----|---------|-----------|
| `pre-deploy` | Version verification | Always |
| `build` | Build distribution packages | If publishing |
| `publish-test` | Publish to Test PyPI | Manual trigger only |
| `publish-pypi` | Publish to PyPI | On release |
| `create-release` | Create GitHub release/tag | Manual with tag |
| `post-deploy` | Deployment summary | After publish |

**Required Secrets:**
- None required (uses OIDC for PyPI)

**Required Environments (GitHub):**
- `pypi`: For production PyPI publishing
- `test-pypi`: For Test PyPI publishing

---

### 3. Dependency Review (`dependency-review.yml`)

**Triggers:**
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Purpose:**
- Reviews dependency changes for security vulnerabilities
- Blocks high-severity vulnerabilities
- Blocks GPL-3.0 and AGPL-3.0 licenses

---

### 4. Nightly Benchmarks (`nightly-benchmark.yml`)

**Triggers:**
- Scheduled daily at 2 AM UTC
- Manual workflow dispatch

**Jobs:**

| Job | Purpose |
|-----|---------|
| `benchmark` | Run JIT, sparse matrix, and log transformation benchmarks |
| `regression-test` | Check for performance degradation |

**Artifacts:**
- `benchmark-results-*`: Benchmark comparison results

---

## Setup Instructions

### 1. Configure PyPI Publishing (OIDC)

The workflows use OpenID Connect (OIDC) for trusted publishing. No tokens needed!

1. Go to your PyPI account settings: https://pypi.org/manage/account/publishing/
2. Add a new publisher with:
   - **PyPI Project Name**: `scptensor`
   - **Owner**: `<your-username>`
   - **Repository name**: `ScpTensor`
   - **Workflow name**: `cd.yml`
   - **Environment name**: `pypi`

3. In GitHub, create environments:
   - Go to Settings > Environments
   - Create `pypi` environment
   - (Optional) Create `test-pypi` environment for Test PyPI

### 2. Enable Coverage Reporting (Optional)

1. Sign up at https://codecov.io/
2. Add your repository
3. Add `CODECOV_TOKEN` as a repository secret
4. Coverage will be automatically uploaded on CI runs

### 3. Configure Branch Protection

1. Go to Settings > Branches
2. Add rule for `main` branch:
   - Require status checks to pass
   - Require branches to be up to date
   - Select required checks: `quality`, `build`, `security`

---

## Usage Examples

### Creating a Release

```bash
# 1. Update version in pyproject.toml
# 2. Commit and push
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin main

# 3. Create a release on GitHub
#    - Go to Actions > CD workflow
#    - Run workflow manually with:
#      - create_git_tag: true
#      - tag_name: v0.2.0
#      - publish_to_pypi: true

# OR create via GitHub CLI:
gh release create v0.2.0 --notes "Release v0.2.0"
```

### Testing PyPI Publishing

```bash
# Go to Actions > CD workflow > Run workflow
# Set:
#   - publish_to_pypi: true
#   - create_git_tag: false
# This will publish to Test PyPI only
```

---

## Matrix Configuration

The CI workflow tests across:

| OS | Python 3.12 | Python 3.13 |
|----|-------------|-------------|
| Ubuntu | ✅ | ✅ |
| macOS | ✅ | ✅ |
| Windows | ✅ | ❌ (excluded to reduce CI time) |

Total: 5 test jobs per run

---

## Caching Strategy

Workflows cache:
- UV package cache (`~/.cache/uv`)
- Virtual environment (`.venv`)
- Cache key includes `pyproject.toml` and `uv.lock` hashes

This typically reduces dependency installation time from 2-3 minutes to 10-30 seconds.

---

## Troubleshooting

### CI fails on Windows/macOS

- Check if issue is OS-specific by looking at individual job logs
- Some tests may need platform-specific skips using `pytest.mark.skipif`

### Mypy errors

- Current config: `strict: false` (will be gradually enabled)
- Add `# type: ignore` for false positives
- Update `pyproject.toml` `[tool.mypy]` section as needed

### Coverage upload fails

- Codecov token is optional; uploads will attempt without it
- Check `CODECOV_TOKEN` secret if using private repo

### PyPI publish fails

- Ensure OIDC is configured correctly in PyPI
- Check version in `pyproject.toml` matches release tag
- Verify `pypi` environment exists in GitHub

---

## Workflow Status Badges

Add these to your `README.md`:

```markdown
[![CI](https://github.com/username/ScpTensor/actions/workflows/ci.yml/badge.svg)](https://github.com/username/ScpTensor/actions/workflows/ci.yml)
[![CD](https://github.com/username/ScpTensor/actions/workflows/cd.yml/badge.svg)](https://github.com/username/ScpTensor/actions/workflows/cd.yml)
[![codecov](https://codecov.io/gh/username/ScpTensor/branch/main/graph/badge.svg)](https://codecov.io/gh/username/ScpTensor)
```

---

**Last Updated:** 2025-01-14
