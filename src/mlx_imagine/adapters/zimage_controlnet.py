"""Adapter for mflux-generate-z-image-controlnet (Z-Image Turbo with the Fun ControlNet Union 2.1). mflux has no
command steps for it. A request gives one or more `control` specs, `type:path[:strength]`.

The statements of ZImageTurboControlnetCommand copy main() of
mflux.models.z_image.cli.z_image_turbo_generate_controlnet. See NOTICE.
tests/test_mflux_drift.py pins that main()."""

from argparse import Namespace

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.z_image.cli import z_image_turbo_generate_controlnet as command
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.controlnet.z_image_turbo_controlnet import ZImageTurboControlnet
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_image import GeneratedImage

DEFAULT_MODEL = "z-image-controlnet"
# The registry key of the one model this command runs. mflux accepts any model that has a ControlNet, but the
# registry gives each model to the command that accepts it, so the adapter must refuse the other models.
MODEL_KEY = "z-image-turbo-controlnet-union-2.1"
IGNORED_OPTIONS = command.IGNORED_OPTIONS
# The repository of the control branch. The download request fetches it with the base model (registry.py).
CONTROLNET_REPO = ModelConfig.z_image_turbo_controlnet_union_2_1().controlnet_model


def build_parser() -> CommandLineParser:
    parser = command.build_parser()
    # The load check needs a request that validate() accepts.
    setattr(parser, "placeholder_options", ["--control", "canny:x.png"])
    setattr(parser, "server_modes", ("controlnet",))
    return parser


class ZImageTurboControlnetCommand:
    latent_creator = ZImageLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # mflux raises ValueError here and calls parser.error() in main(). The server raises ModelConfigError,
        # which the engine turns into an option_invalid answer.
        model_config = ConfigResolution.resolve_restricted(
            args.model, MODEL_KEY, model_path=args.model_path, extra_keys=(), base_model=args.base_model
        )
        if not model_config.controlnet_model:
            raise ModelConfigError(
                f"--model {args.model!r} is not a Union ControlNet model. Use z-image-controlnet (the default)."
            )
        try:
            command.ZImageTurboControlnetCommand._parse_controls(args)
        except ValueError as exc:
            raise ModelConfigError(str(exc)) from None
        return model_config

    @staticmethod
    def load(args: Namespace) -> ZImageTurboControlnet:
        return ZImageTurboControlnet(
            model_config=ZImageTurboControlnetCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            float32=args.float32,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )

    @staticmethod
    def generate(model: ZImageTurboControlnet, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=args.width,
            height=args.height,
            scheduler=args.scheduler,
            num_inference_steps=args.steps,
            controlnet_strength=args.controlnet_strength,
            controls=command.ZImageTurboControlnetCommand._parse_controls(args),
        )
