from model.unet import UNet
from configs import cfg
from model.LERAFormer import LERAFormer
from model.deeplabv3plus import DeepLabV3Plus
from model.segformer_b5 import SegFormer
from model.no_LERA import LERAFormerNoLERA
from model.noLCAF import LERAFormerNoLCAF
from model.STUnet import STUNet
from model.noCECA import LERAFormernoCECA
from model.noDGER import LERAFormernoDGER
from model.NOLSK import LERAFormernolsk
from model.swinunet import SwinUNetBaseline
from model.transunet import TransUNetBaseline
from model.LEFormer import LEFormerBaseline

def build_model(type):
    model = None
    if type == 'unet':
        model = UNet(in_chans=cfg.dataloader.in_channels, num_class=cfg.model.num_classes)
    elif type == 'LERAFormer':
        model = LERAFormer(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'DeepLabV3Plus':
        model = DeepLabV3Plus(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'SegFormer':
        model = SegFormer(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'LERAFormerNoLERA':
        model =LERAFormerNoLERA(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'LERAFormerNoLCAF':
        model =LERAFormerNoLCAF(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'STUNet':
        model = STUNet(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'LERAFormernoCECA':
        model = LERAFormernoCECA(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'LERAFormernoDGER':
        model = LERAFormernoDGER(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'LERAFormernolsk':
        model = LERAFormernolsk(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'SwinUNetBaseline':
        model = SwinUNetBaseline(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'TransUNetBaseline':
        model = TransUNetBaseline(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    elif type == 'LEFormerBaseline':
        model = LEFormerBaseline(in_chans=cfg.dataloader.in_channels, num_classes=cfg.model.num_classes)
    else:
        raise ValueError(f"Unsupported model type: {type}")

    return model