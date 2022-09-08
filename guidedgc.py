import numpy as np
import matplotlib.pyplot as plt

import captum
from captum.attr import visualization as viz

import hydra
from omegaconf import DictConfig

from utils import load_model_pth, ImageLoader, DataTransform


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # load model weights and convert to eval mode
    model = load_model_pth(cfg.model_name, num_classes=cfg.num_classes, model_dir=cfg.model_dir)
    model.eval()

    # load and process image to analyze
    imageLoader = ImageLoader('input/PNEUMONIA/person1946_bacteria_4874.jpeg', transform=DataTransform(image_size=224))
    input_img, original_img = imageLoader.process_images()

    # calculate attribution by Guided Grad-CAM
    guided_gc = captum.attr.GuidedGradCam(model, model.stages[3].blocks[2].drop_path)
    attribution = guided_gc.attribute(
        inputs=input_img,
        target=0
    )
    attribution_img = attribution[0].cpu().permute(1,2,0).detach().numpy()

    # save figures
    figure, _ = viz.visualize_image_attr_multiple(
        attribution_img,
        original_img,
        methods=["heat_map", "original_image"],
        signs = ["absolute_value", "all"],
        fig_size=(15, 15),
        show_colorbar = True
        )
    
    figure.savefig('output/image/ggc_multi.png')


if __name__ == '__main__':
    main()