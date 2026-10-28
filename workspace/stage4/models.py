from stage3.models import DropoutFusionNet
from kinafford.models.variants import build_model as original_build
def build_model(cfg):
 return DropoutFusionNet(cfg) if cfg.get('structured_dropout',False) else original_build(cfg)
