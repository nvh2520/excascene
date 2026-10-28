"""Only dropout-trained fusion extends LiDAR presence; default state/output is unchanged."""
from kinafford.models.variants import build_model as original_build, GeometricFusionNet
class DropoutFusionNet(GeometricFusionNet):
 def sensor_maps(self,batch):
  maps,seg,present=super().sensor_maps(batch)
  if 'lidar' in present and 'lidar_present' in batch:
   presence=batch['lidar_present'].to(present['lidar'])
   present['lidar']=present['lidar']*presence
   maps['lidar']=maps['lidar']*presence[:,None,None,None]
  return maps,seg,present

def build_model(cfg):
 if cfg.get('modality_dropout',0)>0:
  if cfg.get('fusion','geometric')!='geometric':raise ValueError('Dropout registered for geometric fusion only')
  return DropoutFusionNet(cfg)
 return original_build(cfg)
