"""Adapter for mflux-generate-qwen-2.1 (Qwen-Image 2.1 and 2.1 Turbo). mflux has no command steps for it.

The statements of Qwen21Command copy main() of mflux.models.qwen21.cli.qwen21_generate. See NOTICE.
tests/test_mflux_drift.py pins that main()."""

from argparse import Namespace

from mflux.cli.parser.parsers import lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.cli import qwen21_generate
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler, ViggleTurboScheduler
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_image import GeneratedImage

build_parser = qwen21_generate.build_parser
DEFAULT_MODEL = qwen21_generate.DEFAULT_MODEL
FAMILY_MODELS = qwen21_generate.FAMILY_MODELS
IGNORED_OPTIONS = qwen21_generate.IGNORED_OPTIONS


class Qwen21Command:
    latent_creator = Qwen21LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # mflux calls parser.error() in these scheduler checks. validate() has no parser, so the server raises.
        nodes = len(ViggleTurboScheduler.SIGMA_NODES)
        if args.scheduler == "viggle_turbo" and args.steps != nodes:
            raise ModelConfigError(f"--scheduler viggle_turbo samples a fixed schedule; use --steps {nodes}.")
        model_config = ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            base_model=args.base_model,
            extra_keys=FAMILY_MODELS,
        )
        if Qwen21TurboScheduler.is_turbo(model_config):
            if args.guidance is not None and args.guidance != 1:
                raise ModelConfigError(f"Qwen-Image-2.1-Turbo runs without CFG. Use --guidance 1, got {args.guidance}.")
            if args.scheduler == "viggle_turbo":
                raise ModelConfigError("viggle_turbo is the schedule of a LoRA for the base Qwen-Image-2.1.")
        return model_config

    @staticmethod
    def load(args: Namespace) -> QwenImage21:
        return QwenImage21(
            quantize=args.quantize,
            model_path=args.model_path,
            model_config=Qwen21Command.validate(args),
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: QwenImage21, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            negative_prompt=args.negative_prompt or "",
            width=width,
            height=height,
            guidance=args.guidance if args.guidance is not None else 1.0,
            scheduler=args.scheduler,
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            step_cache_ratio=args.step_cache_ratio,
        )
