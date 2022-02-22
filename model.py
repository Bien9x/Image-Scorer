import timm
import tez
import torch
import torch.nn as nn
import config


# class PawpularDataset:
#     def __init__(self, image_paths, dense_features, targets, augmentations):
#         self.image_paths = image_paths
#         self.dense_features = dense_features
#         self.targets = targets
#         self.augmentations = augmentations
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, item):
#         image = cv2.imread(self.image_paths[item])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         if self.augmentations is not None:
#             augmented = self.augmentations(image=image)
#             image = augmented["image"]
#
#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)
#
#         features = self.dense_features[item, :]
#         targets = self.targets[item]
#
#         return {
#             "image": torch.tensor(image, dtype=torch.float),
#             "features": torch.tensor(features, dtype=torch.float),
#             "targets": torch.tensor(targets, dtype=torch.float),
#         }


class AestheticModel(tez.Model):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            config.BACKBONE_MODEL, pretrained=False, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(num_features, 1)

    def forward(self, image, targets=None):
        x = self.backbone(image)
        outputs = self.out(self.dropout(x)).view(-1)

        return outputs, 0, {}
