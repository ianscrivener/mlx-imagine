---
name: imagine-pr
description: Make a clean pull request in mlx-imagine. Covers the checks, the commit, the CodeRabbit question, the push, and the PR body with its CHANGELOG entry. Use when the user asks to open a PR, or to write or update a PR body.
---

# mlx-imagine pull request

## When to use

- The user asks to open a pull request.
- The user asks to write or update a PR body.

## Steps

1. Collect the facts. Use only these sources:
   - `git log --oneline main..HEAD`
   - `git diff --stat main...HEAD`
   - `git diff main...HEAD`
   - If the PR is open: `gh pr view <n> --json title,body,closingIssuesReferences` and `gh pr checks <n>`.
2. Run `just dev-check`. If it fails, stop and report the failure.
3. Make sure that `CHANGELOG.md` has an entry under `## [Unreleased]` for this change. If it has no entry, write one and show it to the user. The `Changelog` CI check fails without it, unless the PR has the label `no-changelog`.
4. If the change affects the API, the options or the config, make sure that `docs/api.md` and `README.md` agree with the code.
5. Commit. Keep each commit to one change. Use the style of the recent commits in `git log`.
6. Write the PR body:
   1. Copy `.github/pull_request_template.md` to `tmp-PR-content.md`. If `tmp-PR-content.md` or `tmp-PR-body.md` exists, ask the user before you replace it.
   2. Put `# Draft PR Content` on line 1 and `Title: <title>` on line 2. Put an empty line after them.
   3. Write the `What` section: one sentence that names the change, then the effect on each group of users. Add `Fixes #<n>` for each linked issue.
   4. Write the `Verification` section. List only the commands that ran, with the result that you saw. If no command ran, write `TODO: add the commands that you ran.`
   5. Do not tick the boxes in the `Checklist` section. The user ticks them.
   6. Apply the `asd-ste100` skill in STE-flavored mode to the text that you wrote. Put that text in a temporary file and run `python3 .agents/skills/asd-ste100/scripts/ste-lint.py <file>`. Fix each hard violation.
7. Ask the user to approve the PR content:
   - Open `tmp-PR-content.md` with `code tmp-PR-content.md`. If that fails, give the user the file path.
   - Tell the user to edit the file, or to ask for changes.
   - Do not continue until the user writes `approve`.
8. Read `tmp-PR-content.md` again after the approval. The user can edit it. Make sure that line 1 is `# Draft PR Content` and line 2 starts with `Title:`.
9. Ask the user: "Do we run CodeRabbit locally first?" If the answer is yes, run `coderabbit review --agent`. Fix the valid findings, and run `just dev-check` again.
10. Ask the user for permission to push. Do not push without it. Push the branch.
11. Send the PR to GitHub. Run this block as one command. For an open PR, change the last line to `gh pr edit <n> --title "$title" --body-file tmp-PR-body.md`.

    ```sh
    title="$(awk 'NR <= 2 && sub(/^Title: */, "") { print; exit }' tmp-PR-content.md)"
    awk 'NR == 1 && /^# Draft PR Content$/ { next } NR == 2 && /^Title:/ { next } { print }' tmp-PR-content.md > tmp-PR-body.md
    gh pr create --title "$title" --body-file tmp-PR-body.md
    ```

    The shell reads the title from the file. Do not type the title into the command: backticks and `$` in a typed title run as shell code.
12. Delete `tmp-PR-content.md` and `tmp-PR-body.md`.

## Rules

- Do not invent facts, numbers or test results.
- Do not push or force-push without the user's approval for each push.
- Before you squash or rewrite the branch history, make a local backup branch: `backup/<branch>-pre-squash-<YYYYMMDD-HHMM>`. Do not push it.
- Do not use `gh pr create --body` or `--fill`. GitHub must get the body that the user approved.
- If `gh` is not available, stop and ask the user.
