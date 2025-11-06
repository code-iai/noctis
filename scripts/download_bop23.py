import os
import os.path as osp
import subprocess
import logging
from functools import partial
import multiprocessing
from tqdm import tqdm

from omegaconf import DictConfig
import hydra

# set level logging
logging.basicConfig(level=logging.INFO)


def huggingface_download_and_unzip(dataset_name: str,
                                   root_dir: str):
    temp_dir = osp.join(root_dir, "tmp")
    zip_dir = osp.join(temp_dir, dataset_name)

    # download the dataset basis, cad models and bop challenge test images
    # Note: Don't download all test images (often > 10GB) and any train images (often > 50GB) per dataset.
    download_cmd = ["hf", "download", "bop-benchmark/{}".format(dataset_name), "--include", "*.zip",
                    "--exclude", "*_train*", "*_val*", "*_all*", "--local-dir", zip_dir, "--repo-type", "dataset"]

    logging.info("Download dataset {}".format(dataset_name))
    subprocess.run(download_cmd, check=True)
    logging.info("Dataset downloaded to {}.".format(zip_dir))

    # unzip all dataset files
    for filename in tqdm(os.listdir(zip_dir)):
        if filename.endswith(".zip"):
            zip_path = osp.join(zip_dir, filename)

            if filename.endswith("_base.zip"):
                # base dataset zip
                unzip_path = root_dir #if dataset_name in ["handal"] else osp.join(root_dir, dataset_name)
            elif filename.endswith("_models.zip"):
                # cad model zip
                unzip_path = osp.join(root_dir, dataset_name, "models")
            elif "onboarding" in filename or "_bop" in filename:
                # onboarding or test images zip
                unzip_path = osp.join(root_dir, dataset_name)
            else:
                # other
                raise ValueError("Unrecognized zip filename: {}".format(filename))

            os.makedirs(unzip_path, exist_ok=True)
            unzip_cmd = ["unzip", "-q", zip_path, "-d", unzip_path]

            logging.info("Unpack file {}".format(filename))
            subprocess.run(unzip_cmd, check=True)
            logging.info("File extracted to {}".format(unzip_path))


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="download")
def download(cfg: DictConfig) -> None:
    bop23_datasets = ["lmo",
                      "tless",
                      "tudl",
                      "icbin",
                      "itodd",
                      "hb",
                      "ycbv"]

    if cfg.dataset_name is None:
        datasets = bop23_datasets
    else:
        datasets = [cfg.dataset_name]

    logging.info("Start downloading the dataset: {}.".format(datasets))

    func = partial(huggingface_download_and_unzip,
                   root_dir=cfg.data.root_dir)

    num_workers = int(cfg.num_workers)
    if num_workers > 0:
        pool = multiprocessing.Pool(processes=num_workers)
        values = list(tqdm(pool.imap_unordered(func, datasets),
                           total=len(datasets)))
    else:
        values = list(tqdm(map(func, datasets),
                           total=len(datasets)))

    logging.info("Finished downloading and unzipping all datasets.")


if __name__ == "__main__":
    download()
