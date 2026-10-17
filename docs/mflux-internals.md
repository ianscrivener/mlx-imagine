# How we touch mflux internals

mlx-imagine uses mflux as a library. This page lists each part of mflux that it calls, reads or changes, and how much risk each part carries. Read it before you upgrade mflux, and after `tests/test_mflux_drift.py` fails.

Levels of risk:

- **High:** a private function (its name starts with `_`), or code that mlx-imagine copies from mflux. Any mflux release can change it.
- **Medium:** public code, but mflux does not document the behavior that mlx-imagine depends on.
- **Low:** the surface that mflux intends callers to use.

"Pinned" means that `tests/test_mflux_drift.py` records a digest or a signature, and fails when mflux changes it.

## 1. Private functions and copied code (high)

| What | Where we use it | Why | Pinned |
|---|---|---|---|
| `CallbackManager._apply_vae_tiling(args, model)` | [engine.py:379](../src/mlx_imagine/engine.py#L379) | Applies `--vae-tiling` and `--vae-tile-size` before each generate. The public `register_callbacks()` also adds a memory saver that frees the text encoder, so mlx-imagine cannot call it. | Yes |
| `CallbackManager._register_memory_saver(args, model)` | [engine.py:383](../src/mlx_imagine/engine.py#L383) | Low-RAM mode only. It frees model parts during a run, so the next job loads the model again. It also sets a 1 GB cache limit for the whole process, which mlx-imagine sets back after the job. | Yes |
| `PathResolution._find_complete_cached_snapshot(repo, patterns)` | [hf_models.py:39](../src/mlx_imagine/hf_models.py#L39) | Tells if the Hugging Face cache holds every file a model loads, before a load starts. | Yes |
| The statements of `mflux-save`'s `main()`, from step 1 on | [engine.py:633](../src/mlx_imagine/engine.py#L633) (`build_model_for_save`) | mflux has no save step that a caller can use. The copy uses `MODEL_CLASSES`, `SaveDispatch.resolve_registry_key`, and Ideogram 4's `resolve_inference_config` and `is_builtin_name`. [NOTICE](../NOTICE) lists the copy. | Yes (digest of `main()`) |
| `ImageUtil._embed_metadata(metadata, path)` and `GeneratedImage._get_metadata()` | [engine.py](../src/mlx_imagine/engine.py) (`save_image`) | The server writes EXIF, and the XMP and IPTC data, one by one, so the settings `image_metadata.exif_enabled` and `iptc_enabled` work. mflux has one switch for all three. | Yes (digests) |
| The statements of `mflux-generate-flux2`'s `main()` | [adapters/flux2.py](../src/mlx_imagine/adapters/flux2.py) | mflux has no command steps for FLUX.2 Klein. The adapter copies the model checks, the model build and the generate call. [NOTICE](../NOTICE) lists the copy. | Yes (digest of `main()`) |
| The statements of `mflux-generate-flux3`'s `main()` | [adapters/flux3.py](../src/mlx_imagine/adapters/flux3.py) | mflux has no command steps for FLUX 3 Image Klein. The adapter copies the model build and the generate call. [NOTICE](../NOTICE) lists the copy. | Yes (digest of `main()`) |

## 2. Undocumented behavior (medium)

| What | Where we use it | What mlx-imagine depends on | Pinned |
|---|---|---|---|
| `CommandLineParser.parse_args()` reads `sys.argv` | [registry.py:133](../src/mlx_imagine/registry.py#L133) | The parser takes no argument list. mlx-imagine sets `sys.argv` under a lock for each parse, then puts it back. `_option_was_provided()` also reads `sys.argv`. | Yes (`_option_was_provided`) |
| `parser.error` replaced | [options.py:121](../src/mlx_imagine/options.py#L121), [engine.py:644](../src/mlx_imagine/engine.py#L644) | A parse error raises an `ApiError` instead of `sys.exit(2)`. | No |
| `parser._actions` (argparse) | [options.py:64](../src/mlx_imagine/options.py#L64) | Maps each request field (the `dest` name) to its flag and its type. | No (argparse is stable) |
| Parser attributes `require_image_paths` and `supports_image_to_image` | [registry.py:163](../src/mlx_imagine/registry.py#L163), [registry.py:171](../src/mlx_imagine/registry.py#L171), [options.py:178](../src/mlx_imagine/options.py#L178) | Tells which models edit an image and which accept an input image. A missing attribute reads as `False`. | No |
| `model.callbacks` lists | [engine.py:586](../src/mlx_imagine/engine.py#L586) (`_unregister`) | `CallbackRegistry` has no unregister call. mlx-imagine removes the low-RAM memory saver from the `before_loop`, `in_loop`, `after_loop` and `interrupt` lists by hand. | No |
| The callback protocol | [engine.py:88](../src/mlx_imagine/engine.py#L88), [engine.py:91](../src/mlx_imagine/engine.py#L91) | `call_before_loop(**kwargs)` and `call_in_loop(t, seed, prompt, latents, config, time_steps)`. mflux calls them by keyword. | No |
| Cancel by `StopImageGenerationException` | [engine.py:86](../src/mlx_imagine/engine.py#L86) | An exception from a callback stops the denoising loop at the next step. mflux's `BatterySaver` uses the same method. The real-weight test showed that it works. | No |
| `model.prompt_cache` | [engine.py:594](../src/mlx_imagine/engine.py#L594) | Qwen, Krea 2 and Ideogram 4 keep one entry for each prompt. mlx-imagine clears the cache at unload, and after a job when it holds more than 8 entries. A model without the cache is skipped. | No |
| `model.tiling_config` | [engine.py:378](../src/mlx_imagine/engine.py#L378) | Saved after load and set back before each job, so the tiling of one job does not carry into the next. | No |
| Constants in the command modules: `DEFAULT_MODEL`, `IGNORED_OPTIONS`, `CONDITIONAL_OPTIONS`, `REJECTED_OPTIONS` | [registry.py:43](../src/mlx_imagine/registry.py#L43), and through `describe_command()` | `DEFAULT_MODEL` decides which command owns a model. The option constants keep ignored options out of the JSON Schemas. | No |
| Lens's `DEFAULT_ENCODER_REPO` | [registry.py:77](../src/mlx_imagine/registry.py#L77) | Lens loads its text encoder from a second repository. A download fetches both. | No |
| `GeneratedImage.image` | [engine.py:400](../src/mlx_imagine/engine.py#L400) | The PIL image, read for the width and the height of the result. | No |

## 3. Intended public surface (low)

| What | Where we use it | Pinned |
|---|---|---|
| The command steps: `validate(args)`, `load(args)`, `generate(model, args, seed, prompt)` | [engine.py:245](../src/mlx_imagine/engine.py#L245), [engine.py:274](../src/mlx_imagine/engine.py#L274), [engine.py:393](../src/mlx_imagine/engine.py#L393), [registry.py:154](../src/mlx_imagine/registry.py#L154) | Yes (signatures, for each stage 1 command) |
| Each command's `build_parser()` | [registry.py](../src/mlx_imagine/registry.py), [options.py](../src/mlx_imagine/options.py) | No |
| `describe_command()` from `mflux-capabilities` | [options.py:151](../src/mlx_imagine/options.py#L151) | No |
| `AVAILABLE_MODELS`, and `ModelConfig.aliases` and `model_name` | [registry.py:208](../src/mlx_imagine/registry.py#L208) | No |
| The weight definitions' `get_download_patterns()` | [registry.py:49](../src/mlx_imagine/registry.py#L49) | No |
| `ModelRevisions.get(repo_id)` | [hf_models.py:102](../src/mlx_imagine/hf_models.py#L102) | No |
| `GeneratedImage.save(path, export_json_metadata, overwrite)` | [engine.py:399](../src/mlx_imagine/engine.py#L399) | Yes (keywords) |
| `model.save_model(path)` | [engine.py](../src/mlx_imagine/engine.py) (`_do_save`) | No |
| The environment variable `MFLUX_CACHE_DIR` | [config.py:315](../src/mlx_imagine/config.py#L315) | No |

## 4. When you upgrade mflux

1. Run `just mflux-bump`, then `just test-drift`.
2. If a pinned digest fails, read mflux's diff of that function. Bring mlx-imagine in line with it. Then record the new digest that the failure prints.
3. Read section 2 again. Those parts have no pin, so a change there shows up only in a real run.
4. Run `just test-real` before a release.
