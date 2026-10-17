"""Commands for mflux models that have no command steps yet.

An adapter module has the shape of an mflux command module: build_parser(), DEFAULT_MODEL, IGNORED_OPTIONS and a
command class with validate(args), load(args) and generate(model, args, seed, prompt). The registry treats it as
any other command. Delete an adapter when mflux ships command steps for the model (ROADMAP.md T-01)."""
