import torch.nn as nn
import torchvision.models as models

INPUT_SIZES = {
    'resnet50': 224,
    'efficientnet_b4': 380,
}


def build_classifier(model_name: str = 'efficientnet_b4', num_classes: int = 2,
                     pretrained: bool = True) -> nn.Module:
    weights_arg = 'IMAGENET1K_V1' if pretrained else None

    if model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet50' or 'efficientnet_b4'.")

    return model
