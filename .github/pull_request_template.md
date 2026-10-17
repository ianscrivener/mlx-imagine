## What

<!-- One sentence that names the change, then the effect on each group of users. Link issues: Fixes #... -->

## Checklist

- [ ] **`just dev-check` passes**
Lint, format, type check, docs lint and the offline tests.<br><br>
- [ ] **Tests added or changed for the new behavior**
Each test docstring starts with `Bug:`. No test loads real weights.<br><br>
- [ ] **CHANGELOG.md has an entry under `## [Unreleased]`**
Or the PR has the label `no-changelog`, for a change that users do not see.<br><br>
- [ ] **Docs agree with the code**
`docs/api.md` and `README.md` for a change to the API, an option or a setting.<br><br>
- [ ] **mflux internals**
A new part of mflux that the code uses has a row in `docs/mflux-internals.md`. A private function or copied code is pinned in `tests/test_mflux_drift.py`.<br><br>
- [ ] **Real-weight check, if necessary**
`just test-real` for a change to the engine, the callbacks or the save flow.

## Verification

<!-- The commands that you ran and the result of each. For a change that affects images, add the images. -->
