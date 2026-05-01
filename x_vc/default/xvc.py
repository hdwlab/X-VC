from omegaconf import OmegaConf
from modelscope import snapshot_download
from huggingface_hub import hf_hub_download
import hydra
import os


def load_xvc(device="cpu"):
    model_dir = snapshot_download(
        'iic/speech_eres2net_sv_en_voxceleb_16k',
        revision="v1.0.3",
    )
    ckpt_path = hf_hub_download(
        'chenxie95/X-VC',
        filename="xvc.pt",
        revision="9e54747d8c4d1ef544b903e2300a4ba040dcc126",
    )

    conf_path = os.path.join(os.path.dirname(__file__), "xvc.yaml")
    cfg = OmegaConf.load(conf_path)
    cfg.model.generator.speaker_encoder.pretrained_dir = model_dir

    model = hydra.utils.instantiate(cfg.model.generator)
    model = model.load_from_checkpoint(
        cfg,
        ckpt_path,
        device=device,
        ema_load=False,
    ).eval()
    return model