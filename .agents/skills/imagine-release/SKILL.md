---
name: imagine-release
description: Prepare a release of mlx-imagine. Covers the version bump, the CHANGELOG section, the lock file, the release PR, and the GitHub Action that tags the release and publishes to PyPI. Use when the user asks to prepare or publish a release.
---

# mlx-imagine release

The GitHub Action `.github/workflows/release.yml` tags the release, makes the GitHub release and publishes to PyPI. Do not do these steps by hand.

## When to use

- The user asks to prepare a release, or to publish one.

## Prepare the release PR

1. Make a branch from an up-to-date `main`: `release/<version>`.
2. Set `version` in `pyproject.toml`. Use Semantic Versioning: a new feature changes the minor number, a fix changes the patch number.
3. Run `uv lock`.
4. In `CHANGELOG.md`, change `## [Unreleased]` to `## [<version>] - <YYYY-MM-DD>`. Add a new empty `## [Unreleased]` section above it.
5. Run `just dev-check`.
6. Ask the user if a real-weight run is necessary. A change to the engine, the callbacks or the save flow needs one. Use the `imagine-testing` skill.
7. Run `just pypi-build`. Make sure that `dist/` has one wheel and one sdist, with the new version in their names.
8. Commit with the message `release: prepare <version>`.
9. Open the PR with the `imagine-pr` skill.

## Publish

1. The user merges the release PR.
2. The user starts the `Release` workflow from `main` in GitHub Actions and types `publish`.
3. The workflow checks that the version has a CHANGELOG section and no tag. It builds, tags `v<version>`, makes the GitHub release from the CHANGELOG section, and waits for approval of the `pypi` environment. Then it publishes through PyPI Trusted Publishing.

## Rules

- Do not tag or publish from your machine.
- Do not upload to PyPI by hand.
- Do not push without the user's approval.

## One-time setup (the user does this once)

- On PyPI, add a pending publisher for the project `mlx-imagine`: workflow `release.yml`, environment `pypi`.
- In GitHub, make the environment `pypi` with a required reviewer, and allow only the `main` branch to deploy.
