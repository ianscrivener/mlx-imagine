"""Adapter for mflux-generate-qwen-2.1-edit (Qwen-Image 2.1 and 2.1 Turbo). mflux has no command steps for it.

The command makes an image from a prompt, and edits with up to 10 reference images, an optional mask and a
strength. The statements of Qwen21EditCommand copy main() of mflux.models.qwen21.cli.qwen21_edit_generate.
See NOTICE. tests/test_mflux_drift.py pins that main()."""

import math
from argparse import Namespace
from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.cli import qwen21_edit_generate
from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler, ViggleTurboScheduler
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import EDIT_SCHEDULERS, MAX_REFERENCES, QwenImage21Edit
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.scale_factor import ScaleFactor

DEFAULT_MODEL = "qwen-image-2.1"
FAMILY_MODELS = qwen21_edit_generate.FAMILY_MODELS
IGNORED_OPTIONS = qwen21_edit_generate.IGNORED_OPTIONS
CONDITIONAL_OPTIONS = qwen21_edit_generate.CONDITIONAL_OPTIONS


def build_parser() -> CommandLineParser:
    parser = qwen21_edit_generate.build_parser()
    setattr(parser, "server_modes", ("txt2img", "edit"))
    setattr(
        parser,
        "server_notes",
        "Without image_paths this model makes an image from the prompt. With image_paths it edits: give 1 to 10 "
        "reference images in order. The prompt refers to them as <image1>, <image2>, and so on. The last image "
        "sets the size of the result when you give no width and height. To keep the size of the first image, "
        "give its width and height (upload_image returns them). Add mask_image to repaint only part of "
        "the first image. strength below 1 keeps more of the first image. To change a person or a style, load "
        "the model with a LoRA that fits Qwen-Image 2.1 (see list_loras and load_model).",
    )
    return parser


def _error(message: str) -> ModelConfigError:
    return ModelConfigError(message)


class Qwen21EditCommand:
    latent_creator = QwenImage21LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # mflux calls parser.error() for these checks. validate() has no parser, so the server raises
        # ModelConfigError, which the engine turns into an option_invalid answer.
        paths = args.image_paths or []
        if args.scheduler not in EDIT_SCHEDULERS:
            raise _error("Qwen-Image-2.1 supports the default linear Euler scheduler or viggle_turbo only.")
        nodes = len(ViggleTurboScheduler.SIGMA_NODES)
        if args.scheduler == "viggle_turbo" and args.steps != nodes:
            raise _error(f"scheduler viggle_turbo samples a fixed schedule. Use steps {nodes}.")
        if len(paths) > MAX_REFERENCES:
            raise _error(f"Qwen-Image-2.1 supports at most {MAX_REFERENCES} reference images.")
        used = [
            name
            for name, on in {
                "mask_image": args.mask_image is not None,
                "auto_mask": args.auto_mask is not None,
                "strength": args.strength != 1.0,
                "enhance_prompt": args.enhance_prompt,
                "verify": args.verify,
            }.items()
            if on
        ]
        if used and not paths:
            raise _error(f"{', '.join(used)} need at least one reference image in image_paths.")
        if args.mask_image is not None and not Path(args.mask_image).exists():
            raise _error(f"mask_image not found: {args.mask_image}")
        if args.auto_mask is not None and not args.auto_mask.strip():
            raise _error("auto_mask must name an object to locate.")
        if not 0 < args.strength <= 1:
            raise _error(f"strength must be in (0, 1], got {args.strength}")
        if args.verify_retries < 0:
            raise _error(f"verify_retries must be 0 or more, got {args.verify_retries}")
        if args.verify_retries and not args.verify:
            raise _error("verify_retries needs verify")
        if args.use_step_cache and not args.use_kv_cache:
            raise _error("use_step_cache needs use_kv_cache (the default)")
        guidance = args.guidance if args.guidance is not None else 1.0
        if not math.isfinite(guidance) or guidance < 1:
            raise _error("guidance must be finite and at least 1.")
        try:
            width, height = Qwen21EditCommand._size(args, paths)
            QwenImage21LatentCreator.validate(
                width if width is not None else 32, height if height is not None else 32, args.steps, len(paths)
            )
            QwenImage21LatentCreator.validate_resolution(args.output_resolution)
            model_config = ConfigResolution.resolve_restricted(
                args.model,
                DEFAULT_MODEL,
                model_path=args.model_path,
                extra_keys=FAMILY_MODELS,
                base_model=args.base_model,
            )
        except ValueError as exc:
            raise _error(str(exc)) from None
        if Qwen21TurboScheduler.is_turbo(model_config):
            if args.guidance is not None and args.guidance != 1:
                raise _error(f"Qwen-Image-2.1-Turbo runs without CFG. Use guidance 1, got {args.guidance}.")
            if args.scheduler == "viggle_turbo":
                raise _error("viggle_turbo is the schedule of a LoRA for the base Qwen-Image-2.1.")
        return model_config

    @staticmethod
    def _size(args: Namespace, paths: list) -> tuple:
        width, height = args.width, args.height
        if isinstance(width, ScaleFactor) or isinstance(height, ScaleFactor):
            width, height = DimensionResolver.resolve(
                width=width if width is not None else ScaleFactor(1),
                height=height if height is not None else ScaleFactor(1),
                reference_image_path=paths[-1] if paths else None,
            )
        return width, height

    @staticmethod
    def load(args: Namespace) -> QwenImage21Edit:
        return QwenImage21Edit(
            quantize=args.quantize,
            model_path=args.model_path,
            model_config=Qwen21EditCommand.validate(args),
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: QwenImage21Edit, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        paths = args.image_paths or []
        width, height = Qwen21EditCommand._size(args, paths)
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            negative_prompt=args.negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=args.steps,
            guidance=args.guidance if args.guidance is not None else 1.0,
            image_paths=paths,
            output_resolution=args.output_resolution,
            use_kv_cache=args.use_kv_cache,
            mask_image=args.mask_image,
            auto_mask=args.auto_mask,
            strength=args.strength,
            use_step_cache=args.use_step_cache,
            step_cache_threshold=args.step_cache_threshold,
            enhance_prompt=args.enhance_prompt,
            verify=args.verify,
            verify_retries=args.verify_retries,
            scheduler=args.scheduler,
        )
