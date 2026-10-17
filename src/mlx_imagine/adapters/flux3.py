"""Adapter for mflux-generate-flux3 (FLUX 3 Image Klein). mflux has no command steps for it.

The statements of Flux3KleinCommand copy main() of mflux.models.flux3.cli.flux3_generate. See NOTICE.
tests/test_mflux_drift.py pins that main()."""

from argparse import Namespace

from mflux.cli.parser.parsers import lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux3.cli import flux3_generate
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.variants import Flux3Klein
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_image import GeneratedImage

build_parser = flux3_generate.build_parser
DEFAULT_MODEL = flux3_generate.DEFAULT_MODEL
FAMILY_MODELS = flux3_generate.FAMILY_MODELS
IGNORED_OPTIONS = flux3_generate.IGNORED_OPTIONS
CONDITIONAL_OPTIONS = flux3_generate.CONDITIONAL_OPTIONS
REJECTED_OPTIONS = flux3_generate.REJECTED_OPTIONS


class Flux3KleinCommand:
    latent_creator = Flux3LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # mflux checks the option on sys.argv. validate() runs after the parser, so the server checks the value.
        if args.negative_prompt:
            raise ModelConfigError("--negative-prompt is not supported for FLUX 3 Image Klein.")
        return ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )

    @staticmethod
    def load(args: Namespace) -> Flux3Klein:
        return Flux3Klein(
            model_config=Flux3KleinCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
        )

    @staticmethod
    def generate(model: Flux3Klein, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=None)
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=args.steps,
            guidance=args.guidance,
        )
