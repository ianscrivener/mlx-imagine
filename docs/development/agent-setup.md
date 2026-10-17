# Agent setup

This page tells where the files for coding agents (Claude Code, Cursor, Codex and others) go, and which files the repository shares.

## Shared files and personal files

| Scope | Files | In git |
|---|---|---|
| Shared, this project | `AGENTS.md`, `.agents/skills/`, the symlinks in `.claude/skills/` and `.cursor/skills/` | Yes |
| Personal, this project | `CLAUDE.md`, `CLAUDE.local.md`, `.claude/settings.local.json`, `.claude/agents/`, `.mcp.json` | No (in `.gitignore`) |
| Personal, all projects | `~/.claude/CLAUDE.md`, `~/.claude/skills/`, `~/.claude/settings.json` | Not in the repository |

- Put the shared agent rules in `AGENTS.md`. The repository ignores `CLAUDE.md`, so each developer can keep personal rules there.
- To share a file that `.gitignore` ignores, add a `!` rule for it to `.gitignore`.

## Skills

`.agents/skills/` holds the skills. It does not belong to one tool. `.claude/skills/<name>` and `.cursor/skills/<name>` are symlinks to `.agents/skills/<name>`, so each tool reads the same file.

To add a skill:

1. Make `.agents/skills/<name>/SKILL.md`, with `name` and `description` in its front matter.
2. Make the two symlinks:

   ```sh
   ln -s ../../.agents/skills/<name> .claude/skills/<name>
   ln -s ../../.agents/skills/<name> .cursor/skills/<name>
   ```

3. Add the skill to the table in `AGENTS.md`.
4. Write the skill in strict ASD-STE100 and run `just dev-docs`.

The GitNexus skills (`gitnexus-*`) are an exception. `gitnexus analyze` writes them, and `.gitignore` ignores them.

## Personal scratch files

Add a personal file that is not an agent file to `.git/info/exclude`. Only your machine reads that file. For example: `z*.*`.
