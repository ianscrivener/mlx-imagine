"""Adapter for mflux-generate-qwen-2.1-controlnet (Qwen-Image 2.1 with the Fun ControlNet Union). mflux has no
command steps for it. One checkpoint reads a control image (canny, depth, pose and more), and the same branch
inpaints when the request has a source image and a mask.

The statements of Qwen21ControlnetCommand copy main() of mflux.models.qwen21.cli.qwen21_controlnet_generate.
See NOTICE. tests/test_mflux_drift.py pins that main()."""

from argparse import Namespace
from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.cli import qwen21_controlnet_generate
from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler
from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_image import GeneratedImage

DEFAULT_MODEL = qwen21_controlnet_generate.DEFAULT_MODEL
FAMILY_MODELS = qwen21_controlnet_generate.FAMILY_MODELS
IGNORED_OPTIONS = qwen21_controlnet_generate.IGNORED_OPTIONS
CONDITIONAL_OPTIONS = qwen21_controlnet_generate.CONDITIONAL_OPTIONS
# The repository of the control branch. The download request fetches it with the base model (registry.py).
CONTROLNET_REPO = ModelConfig.qwen_image_21_controlnet_union().controlnet_model


def build_parser() -> CommandLineParser:
    parser = qwen21_controlnet_generate.build_parser()
    # The load check needs a request that validate() accepts: a control image, or an image and a mask.
    setattr(parser, "placeholder_options", ["--controlnet-image-path", "x.png"])
    setattr(parser, "server_modes", ("controlnet", "inpaint"))
    setattr(
        parser,
        "server_notes",
        "Three ways to use this model. (1) Control: give controlnet_image_path, a ready control image (canny, "
        "depth, grayscale, HED, lineart, MLSD, pose or scribble: one checkpoint reads all, so give no type). "
        "(2) Inpaint: give image_path and mask_image. White in the mask is drawn again. "
        "(3) Both: give all three, to inpaint under control. controlnet_strength sets how strong the control is. "
        "The server does not make control images: make them first. Send each image with upload_image and use its "
        "upload_id. Call preview_mask to check a mask. The first load takes long and uses much memory.",
    )
    return parser


class Qwen21ControlnetCommand:
    latent_creator = QwenImage21LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # mflux raises ValueError here, and calls parser.error() in the Turbo check. The server raises
        # ModelConfigError, which the engine turns into an option_invalid answer.
        model_config = ConfigResolution.resolve_restricted(
            args.model, DEFAULT_MODEL, model_path=args.model_path, extra_keys=FAMILY_MODELS, base_model=args.base_model
        )
        try:
            if args.controlnet_image_path is None and args.image_path is None:
                raise ValueError("Give controlnet_image_path, or image_path with mask_image to inpaint.")
            if (args.image_path is None) != (args.mask_image is None):
                raise ValueError("Inpainting needs both image_path and mask_image.")
            for name in ("width", "height"):
                value = getattr(args, name)
                if value is not None and (not isinstance(value, int) or value < 32 or value % 32):
                    raise ValueError(f"{name} must be a positive multiple of 32.")
            QwenImage21LatentCreator.validate_resolution(args.output_resolution)
            if Path(args.output).suffix.lower() not in (".png", ".webp", ".tif", ".tiff"):
                raise ValueError("Qwen-Image-2.1 outputs RGBA; use PNG, WebP or TIFF to retain transparency.")
        except ValueError as exc:
            raise ModelConfigError(str(exc)) from None
        if Qwen21TurboScheduler.is_turbo(model_config):
            if args.guidance is not None and args.guidance != 1:
                raise ModelConfigError(f"Qwen-Image-2.1-Turbo runs without CFG. Use guidance 1, got {args.guidance}.")
            if args.scheduler == "viggle_turbo":
                raise ModelConfigError("viggle_turbo is the schedule of a LoRA for the base Qwen-Image-2.1.")
        # generate_image() raises ValueError for this, after the job started. Refuse it now.
        if args.guidance is not None and not 1 <= args.guidance < float("inf"):
            raise ModelConfigError("guidance must be finite and at least 1.")
        return model_config

    @staticmethod
    def load(args: Namespace) -> QwenImage21Controlnet:
        return QwenImage21Controlnet(
            model_config=Qwen21ControlnetCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: QwenImage21Controlnet, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            controlnet_image_path=args.controlnet_image_path,
            controlnet_strength=args.controlnet_strength,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=1.0 if args.guidance is None else args.guidance,
            negative_prompt=args.negative_prompt,
            output_resolution=args.output_resolution,
            image_path=args.image_path,
            mask_image=args.mask_image,
        )
