"""Adapter for mflux-generate-flux2 (FLUX.2 Klein, all sizes). mflux has no command steps for it.

The statements of Flux2KleinCommand copy main() of mflux.models.flux2.cli.flux2_generate. See NOTICE.
tests/test_mflux_drift.py pins that main()."""

from argparse import Namespace

from mflux.cli.parser.parsers import lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux2.cli import flux2_generate
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2Klein
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_image import GeneratedImage

build_parser = flux2_generate.build_parser
DEFAULT_MODEL = flux2_generate.DEFAULT_MODEL
FAMILY_MODELS = flux2_generate.FAMILY_MODELS
CONDITIONAL_OPTIONS = flux2_generate.CONDITIONAL_OPTIONS
REJECTED_OPTIONS = flux2_generate.REJECTED_OPTIONS
IGNORED_OPTIONS: dict[str, str] = {}


class Flux2KleinCommand:
    latent_creator = Flux2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # mflux checks the option on sys.argv. validate() runs after the parser, so the server checks the value.
        if args.negative_prompt:
            raise ModelConfigError("--negative-prompt is not supported for FLUX.2.")
        model_config = ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )
        named = args.model_path is not None and args.base_model is None
        named = (
            named and ConfigResolution.infer_family_member(args.model_path, DEFAULT_MODEL, FAMILY_MODELS) is not None
        )
        judged = args.model_path is None or args.base_model is not None or named
        guidance = 1.0 if args.guidance is None else args.guidance
        if judged and guidance != 1.0 and "base" not in model_config.model_name.lower():
            raise ModelConfigError(
                "--guidance is only supported for FLUX.2 base models. Use --guidance 1.0, "
                "or declare a base checkpoint with --base-model."
            )
        return model_config

    @staticmethod
    def load(args: Namespace) -> Flux2Klein:
        return Flux2Klein(
            model_config=Flux2KleinCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: Flux2Klein, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=1.0 if args.guidance is None else args.guidance,
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            scheduler="flow_match_euler_discrete",
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )
