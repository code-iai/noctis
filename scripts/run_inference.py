import logging
import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader

if True:
    # set random number generator seed
    import random
    import numpy
    import torch
    random.seed(2025)
    numpy.random.seed(2025)
    torch.manual_seed(2025)

# set level logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="run_inference")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    logging.info("Training script. The outputs of hydra will be stored in: {}".format(output_path))
    logging.info("Initializing logger, callbacks and trainer")

    # create trainer (used because of the 'LightningModule' interface of detector model)
    trainer = instantiate(cfg.machine.trainer)

    # create query/test and reference dataset
    default_query_dataloader_config = cfg.data.query_dataloader
    default_ref_dataloader_config = cfg.data.reference_dataloader

    query_dataloader_config = default_query_dataloader_config.copy()
    ref_dataloader_config = default_ref_dataloader_config.copy()

    if cfg.dataset_name in ["hb", "tless"]:
        query_dataloader_config.split = "test_primesense"
    else:
        query_dataloader_config.split = "test"
    query_dataloader_config.root_dir += str(cfg.dataset_name)
    query_dataset = instantiate(query_dataloader_config)

    query_dataloader = DataLoader(query_dataset,
                                  batch_size=1,  # only support a single image for now
                                  num_workers=cfg.machine.num_workers,
                                  shuffle=False)

    # create template dataset
    if cfg.model.onboarding_config.rendering_type == "onboarding_static":
        logging.info("Using static onboarding images")
        ref_dataloader_config.template_dir += osp.join(cfg.dataset_name, "onboarding_static")
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "onboarding_dynamic":
        logging.info("Using dynamic onboarding images")
        ref_dataloader_config.template_dir += osp.join(cfg.dataset_name, "onboarding_dynamic")
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "templates":
        ref_dataloader_config.template_dir += osp.join(cfg.template_dir, cfg.dataset_name)
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "pbr":
        logging.info("Using BlenderProc for reference images")
        ref_dataloader_config._target_ = "src.dataloader.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = query_dataloader_config.root_dir
        ref_dataloader_config.template_dir += osp.join(cfg.template_dir, cfg.dataset_name)
        ref_dataloader_config.split = "train_pbr"
        ref_dataloader_config.reset_metadata = query_dataloader_config.reset_metadata
        os.makedirs(ref_dataloader_config.template_dir, exist_ok=True)  # create, if missing
        ref_dataset = instantiate(ref_dataloader_config)
    else:
        raise NotImplementedError("Unknown template rendering type of name: <{}>".format(cfg.model.onboarding_config.rendering_type))

    # create 'NOCTIS' detector model
    logging.info("Initializing model")
    model = instantiate(cfg.model)
    model.dataset_name = cfg.dataset_name
    model.ref_dataset = ref_dataset

    if model.name_prediction_file is None:
        segmentation_name = cfg.model.segmentor_model._target_.split(".")
        if segmentation_name[-1][0].islower():
            # likely a function name
            segmentation_name = segmentation_name[-2]
        else:
            # should be class name
            segmentation_name = segmentation_name[-1]
        rendering_type = cfg.model.onboarding_config.rendering_type
        level_template = cfg.model.onboarding_config.level_templates
        semantic_agg_function = cfg.model.matching_config.semantic_aggregation_function \
            if cfg.model.matching_config.semantic_score_weight_factor > 0 else "None"
        appearance_agg_function = cfg.model.matching_config.appearance_aggregation_function \
            if cfg.model.matching_config.appearance_score_weight_factor > 0 else "None"
        model.name_prediction_file = "result_{}_{}_{}{}_sem{}_appe{}".format(
            cfg.dataset_name, segmentation_name, rendering_type, level_template,
            semantic_agg_function, appearance_agg_function)

    logging.info("Loading dataloader for {} done!".format(cfg.dataset_name))
    trainer.test(model, dataloaders=query_dataloader)
    logging.info("---" * 20)


if __name__ == "__main__":
    run_inference()
