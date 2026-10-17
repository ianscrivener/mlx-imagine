---
name: imagine-testing
description: Run the mlx-imagine tests. Covers the offline tests (fake models, no weights), the mflux drift test, and the real-weight checks under the memory watchdog. Use to run tests, to fix a failing test, after an mflux upgrade, or to check a change with real weights.
---

# mlx-imagine testing

mlx-imagine has three kinds of test:

| Kind | Recipe | Loads weights | When |
|---|---|---|---|
| Offline tests | `just test` | No | After each change |
| Drift test | `just test-drift` | No | After each mflux upgrade |
| Real-weight checks | `just test-real` | Yes | Before a release, and after a change to the engine, the callbacks or the save flow |

## When to use

- You run tests, or a test fails.
- `just test-drift` fails after an mflux upgrade.
- You must check a change with real model weights.

## Offline tests

1. Run `just test`. Use `just test-k <expression>`, `just test-file <name>` or `just test-debug` to run fewer tests.
2. No test downloads a model or loads weights. `tests/conftest.py` sets the Hugging Face Hub offline before mflux loads. The fake model is in `tests/_fake_command.py`.
3. Each test docstring starts with `Bug:` and names the bug that makes the test fail.
4. Warnings are errors. Expect a warning with `pytest.warns`. Do not add a general ignore.
5. If a change affects the API, the options, the config or the job flow, add a test for the changed behavior.
6. If a test fails, report the test name and the main assertion line.

## Drift test

`tests/test_mflux_drift.py` records a digest of each private mflux function that mlx-imagine calls or copies. It also pins the signatures of the command steps. [docs/mflux-internals.md](../../../docs/mflux-internals.md) lists each one.

If the drift test fails:

1. Read the mflux diff of the function that the failure names.
2. Change mlx-imagine to agree with the new mflux code.
3. Record the new digest that the failure prints.
4. Do not record a new digest before you do step 2.

## Real-weight checks

A real model can use most of the memory of the machine. A load once stopped the Mac.

1. Ask the user before each real-weight run. Do not start one without approval.
2. Tell the user to close the apps that use much memory, for example a web browser with many tabs.
3. Report the memory before the run: `memory_pressure | tail -1` and `sysctl vm.swapusage`.
4. Run `just test-real`. Give `steps` and `model` to run fewer checks, for example `just test-real "load,seeds,unload"`.
   - The steps are `load`, `seeds`, `cancel`, `img2img`, `mcp`, `lowram`, `save` and `unload`.
   - The default model is the saved model `z-image-turbo-q8` in `mflux_models.dir` of `config.yaml`.
   - `just test-real-save` saves and quantizes `z-image-turbo` from bf16 weights, about 20 GB. It then deletes the save.
5. `scripts/real/run.sh` starts the server under a watchdog. The watchdog stops the server and its child processes when free memory is less than 5%, or when swap grows by more than 10 GB.
6. Read the result with `just test-real-log`. Each check gives one `PASS` or `FAIL` line. Report the lowest free memory and the swap that the watchdog recorded.
7. Show the user the output images in `output/`. Ask the user to look at them. Do not judge image quality yourself from the file size.
8. Do not commit the output images or the logs.

## Rules

- Do not run real weights in a pytest test.
- Do not make a test pass by a change to its assertion. Find the cause.
- Do not delete a test that fails. Ask the user first.
