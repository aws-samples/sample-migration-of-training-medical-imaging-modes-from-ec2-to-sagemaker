"""
Model definitions for medical image segmentation
"""
from monai.networks.nets import SegResNet, SwinUNETR


class ModelDefinition:
    """Factory class for creating segmentation models"""
    
    def __init__(self, model_name="SegResNet"):
        self.model_name = model_name
    
    def get_model(self):
        """Return the specified model architecture"""
        if self.model_name == "SegResNet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                init_filters=16,
                dropout_prob=0.2
            )
        elif self.model_name == "SwinUNETR":
            return SwinUNETR(
                img_size=(128, 128, 64),
                in_channels=1,
                out_channels=1,
                feature_size=24,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
