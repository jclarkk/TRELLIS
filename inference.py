import os

os.environ['SPCONV_ALGO'] = 'native'  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.
import argparse
import torch
import uuid
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def run(args):
    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    # Load images
    images = []
    for image_path in args.image_paths:
        image = Image.open(image_path)
        images.append(image)

    # Run the pipeline
    if len(images) > 1:
        outputs = pipeline.run_multi_image(
            images,
            seed=args.seed,
            formats=['mesh', 'gaussian']
        )
    elif len(images) == 1:
        outputs = pipeline.run(
            images[0],
            seed=args.seed,
            formats=['mesh', 'gaussian']
        )
    else:
        raise ValueError('No images provided')

    del pipeline
    torch.cuda.empty_cache()

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.90,  # Ratio of triangles to remove in the simplification process
        texture_size=2048,  # Size of the texture used for the GLB
        gs_renderer='gsplat',  # Renderer to use for the Gaussian representation
    )

    # Use image file name as output name
    if len(args.image_paths) == 1:
        output_name = os.path.splitext(os.path.basename(args.image_paths[0]))[0]
    else:
        output_name = str(uuid.uuid4()).replace('-', '')
    glb.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run(args)