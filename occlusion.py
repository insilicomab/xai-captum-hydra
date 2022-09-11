import captum
from captum.attr import visualization as viz

import hydra
from omegaconf import DictConfig

from utils import load_model_pth, ImageLoader, DataTransform


@hydra.main(version_base=None, config_path='config', config_name='occlusion')
def main(cfg: DictConfig):

    print(f'target: {cfg.target}')

    # load model weights and convert to eval mode
    model = load_model_pth(cfg.model_name, num_classes=cfg.num_classes, model_dir=cfg.model_dir)
    model.eval()

    # load and process image to analyze
    imageLoader = ImageLoader(cfg.input_img_dir, transform=DataTransform(config=cfg))
    input_img, original_img = imageLoader.process_images()

    # calculate attribution by Occlusion
    occlusion = captum.attr.Occlusion(model)

    attribution = occlusion.attribute(
        inputs=input_img,
        target=cfg.target,
        sliding_window_shapes=tuple(cfg.occlusion.sliding_window_shapes),
        strides=tuple(cfg.occlusion.strides),
        baselines=cfg.occlusion.baselines,
        additional_forward_args=cfg.occlusion.additional_forward_args,
        perturbations_per_eval=cfg.occlusion.perturbations_per_eval,
        show_progress=cfg.occlusion.show_progress
    )

    attribution_img = attribution[0].cpu().permute(1,2,0).detach().numpy()

    # save a figure
    if cfg.vis_img.enable:
        figure, _ = viz.visualize_image_attr(
            attribution_img,
            original_img,
            method=cfg.vis_img.method,
            sign=cfg.vis_img.sign,
            plt_fig_axis=cfg.vis_img.plt_fig_axis,
            outlier_perc=cfg.vis_img.outlier_perc,
            cmap=cfg.vis_img.cmap,
            alpha_overlay=cfg.vis_img.alpha_overlay,
            show_colorbar=cfg.vis_img.show_colorbar,
            title=cfg.vis_img.title,
            fig_size=cfg.vis_img.fig_size,
            use_pyplot=cfg.vis_img.use_pyplot
        )

        figure.savefig(cfg.output_img_dir)

    # save multiple figures
    if cfg.vis_img_multi.enable:
        figure_m, _ = viz.visualize_image_attr_multiple(
            attribution_img,
            original_img,
            methods=cfg.vis_img_multi.methods,
            signs=cfg.vis_img_multi.signs,
            outlier_perc=cfg.vis_img_multi.outlier_perc,
            cmap=cfg.vis_img_multi.cmap,
            alpha_overlay=cfg.vis_img_multi.alpha_overlay,
            show_colorbar=cfg.vis_img_multi.show_colorbar,
            titles=cfg.vis_img_multi.titles,
            fig_size=cfg.vis_img_multi.fig_size,
            use_pyplot=cfg.vis_img_multi.use_pyplot,
            )
    
        figure_m.savefig(cfg.output_multi_img_dir)


if __name__ == '__main__':
    main()