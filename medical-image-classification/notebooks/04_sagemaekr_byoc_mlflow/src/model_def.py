from monai.networks.nets import DenseNet121, ViT
import torch

class ModelDef:
    def __init__(self, num_classes, model_name):
        self.num_classes = num_classes
        self.model_name = model_name

    def get_model(self):
        if self.model_name == "DenseNet121":
            return DenseNet121(spatial_dims=2, in_channels=1, out_channels=self.num_classes)
        
        elif self.model_name == "ViT":
            return ViT(
                in_channels=1,
                img_size=(256, 256, 1),
                patch_size=(16, 16, 1),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                classification=True,
                num_classes=self.num_classes,
            )   
        
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        
def main():
    model_def = ModelDef(num_classes=8, model_name='ViT')
    model = model_def.get_model()
    print(model)
    
    a = torch.randn(1, 1, 256, 256, 1)
    b = model(a)
    print(b[0].shape)
    print(b[1][0].shape) 
    print(b[1][1].shape)
    print(b[1][2].shape)
    print(b[1][3].shape)
    
if __name__ == "__main__":
    main()
