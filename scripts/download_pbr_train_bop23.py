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

    # download the PBR training data
    # Note: LM-O is subset of LM and does not have its own PBR training data
    download_cmd = ["hf", "download", "bop-benchmark/{}".format(dataset_name if dataset_name != "lmo" else "lm"),
                    "--include", "*_train_pbr.zip", "--local-dir", zip_dir, "--repo-type", "dataset"]

    logging.info("Download PBR training data of dataset {}".format(dataset_name))
    subprocess.run(download_cmd, check=True)
    logging.info("PBR training data downloaded to {}.".format(zip_dir))

    # unzip all dataset files
    for filename in tqdm(os.listdir(zip_dir)):
        if filename.endswith("_train_pbr.zip"):
            zip_path = osp.join(zip_dir, filename)
            unzip_path = osp.join(root_dir, dataset_name)

            os.makedirs(unzip_path, exist_ok=True)
            unzip_cmd = ["unzip", "-q", zip_path, "-d", unzip_path]

            logging.info("Unpack file {}".format(filename))
            #subprocess.run(unzip_cmd, check=True)
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

    logging.info("Start downloading the PBR training data of dataset: {}.".format(datasets))

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

    logging.info("Finished downloading and unzipping the PBR training data of all datasets.")


if __name__ == "__main__":
    download()
