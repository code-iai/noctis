import os
import os.path as osp
import logging
import requests
from tqdm import tqdm

from huggingface_hub import hf_hub_download

from omegaconf import DictConfig
import hydra

# set level logging
logging.basicConfig(level=logging.INFO)


def download_file(url: str,
                  file_path: str,
                  force_reload: bool = False,
                  block_size: int = 1024,
                  reraise_error: bool = True) -> bool:
    """
    Downloads a file from the given URL and saves it to the specified path.

    :param str url: The URL of the file.
    :param str file_path: The path where the file will be saved.
    :param bool force_reload: Downloads the file even if it already exists.
    :param int block_size: The block size in bytes to read data from the stream.
    :param bool reraise_error: In case of a failure the caught error is reraised instead returning 'False'
    :return: 'True' if the file was successfully downloaded, 'False' otherwise.
    :rtype: bool
    :raises FileNotFoundError: If the file cannot be downloaded or saved.
    :raises Exception: If an unexpected error occurs during the download process.
    """
    try:
        if not force_reload and os.path.exists(file_path):
            # check if the file already exists
            logging.info("{} already exists! Skipping download.".format(file_path))
            return True

        # create directory
        os.makedirs(osp.dirname(file_path), exist_ok=True)

        logging.info("Attempting to download a PyTorch checkpoint from: {}".format(url))

        # # download file and save it in the specified path
        # command = ["wget", "--no-check-certificate", "-O", file_path, url]
        # subprocess.call(command)

        # send request to download file
        response = requests.get(url, stream=True)

        # raise an 'HTTPError' for bad responses
        response.raise_for_status()

        # read received data size in byts
        total_size = int(response.headers.get("content-length", 0))

        # block-wise read and save the data as file to the specified path
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
        with open(file_path, 'wb') as file:
            for data in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

        logging.info("File successful downloaded and saved to: {}".format(file_path))

        return True
    except FileNotFoundError as e:
        logging.error("Error: Checkpoint file not found: {}".format(e))

        if reraise_error:
            # reraise the error
            raise e
        else:
            return False


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="download")
def download(cfg: DictConfig) -> None:
    checkpoint_dir = osp.join(cfg.machine.root_dir, "checkpoints")

    logging.info("Start downloading and saving the checkpoints of all sub model in: {}".format(checkpoint_dir))

    # DINOv2 'https://github.com/facebookresearch/dinov2' dinov2_vitl14_reg
    dinov2_model_dict = {"dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",             # 88 MB
                         "dinov2_vits14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",    # 88 MB
                         "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",             # 346 MB
                         "dinov2_vitb14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",    # 346 MB
                         "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",             # 1.2 GB
                         "dinov2_vitl14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",    # 1.2 GB
                         "dinov2_vitg14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",             # 4.5 GB
                         "dinov2_vitg14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"}    # 4.5 GB
    selected_dinov2_model_name = "dinov2_vitl14"  # default segmentation model used in CNOS

    for (model_name, model_url) in dinov2_model_dict.items():
        if selected_dinov2_model_name is not None and selected_dinov2_model_name != model_name:
            continue

        checkpoint_url = dinov2_model_dict[model_name]
        checkpoint_file_path = osp.join(checkpoint_dir, "dinov2", checkpoint_url.split("/")[-1])
        download_file(checkpoint_url, checkpoint_file_path)

    # GroundingDINO 'https://github.com/IDEA-Research/GroundingDINO'
    gd_model_dict = {"groundingdino_t": "groundingdino_swint_ogc.pth",         # 694 MB
                     "groundingdino_b": "groundingdino_swinb_cogcoor.pth"}     # 938 MB
    selected_gd_model_name = None   #"groundingdino_t"

    for (model_name, checkpoint_filename) in gd_model_dict.items():
        if selected_gd_model_name is not None and selected_gd_model_name != model_name:
            continue

        local_checkpoint_dir = osp.join(checkpoint_dir, "gdino")
        hf_hub_download(repo_id="ShilongLiu/GroundingDINO",
                        filename=checkpoint_filename,
                        local_dir=local_checkpoint_dir)

    # SAM 'https://github.com/facebookresearch/sam2'
    sam_model_dict = {"vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # 375 MB
                      "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # 1.2 GB
                      "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"}  # 2.6 GB
    selected_sam_model_name = None  # "vit_h"  # default segmentation model used in CNOS

    for (model_name, model_url) in sam_model_dict.items():
        if selected_sam_model_name is not None and selected_sam_model_name != model_name:
            continue

        checkpoint_url = sam_model_dict[model_name]
        checkpoint_file_path = osp.join(checkpoint_dir, "sam", checkpoint_url.split("/")[-1])
        download_file(checkpoint_url, checkpoint_file_path)

    # FastSAM 'https://github.com/CASIA-IVA-Lab/FastSAM'
    # Note: Google Drive files need multiple get request for cookies etc., so they cannot be downloaded directly.
    # Therefore, use 'gdown' as helper.
#    fs_model_id = "1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv"     # default model
#    fs_checkpoint_file_path = osp.join(checkpoint_dir, "FastSAM", "FastSAM-x.pt")
#    if not osp.exists(fs_checkpoint_file_path):
#        fs_command = "gdown --no-cookies --no-check-certificate -O {} {}".format(fs_checkpoint_file_path, fs_model_id)
#        os.system(fs_command)

    # MobileSAM 'https://github.com/ChaoningZhang/MobileSAM'
    ms_checkpoint_url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    ms_checkpoint_file_path = osp.join(checkpoint_dir, "mobilesam", "mobile_sam.pt")
    download_file(ms_checkpoint_url, ms_checkpoint_file_path)

    logging.info("Finished downloading all checkpoints.")


if __name__ == "__main__":
    download()
