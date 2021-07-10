from torch import nn
from torchvision import models
import timm


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    model_type = config.model.model_type
    num_classes = config.dataset.num_of_classes

    if model_type == 'resnet34':
        print("ResNet34")
        model = timm.create_model('resnet34', pretrained=True,
            drop_rate=config.model.drop_rate)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif 'efficientnet' in model_type:
        print(f"{model_type} using Dropout {config.model.drop_rate}")
        model = timm.create_model(f'{model_type}', pretrained=True,
            drop_path_rate=0.2,
            drop_rate=config.model.drop_rate)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_type == 'torchvision_resnet34':
        print("torchvision ResNet34")
        print(f"{model_type} using Dropout {config.model.drop_rate}")
        model = models.resnet34(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_type == 'torchvision_resnet18':
        print("torchvision ResNet18")
        print(f"{model_type} using Dropout {config.model.drop_rate}")
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_type == 'torchvision_resnet18_antialias':
        print("torchvision antialising ResNet18")
        print(f"{model_type} using Dropout {config.model.drop_rate}")
        # model = models.resnet18(pretrained=True)
        import antialiased_cnns
        model = antialiased_cnns.resnet18(pretrained=True) 
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif model_type == 'torchvision_resnet50':
        print("torchvision ResNet50")
        print(f"{model_type} using Dropout {config.model.drop_rate}")
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif model_type == 'torchvision_resnet34_2fc':
        print("torchvision ResNet34")
        print(f"{model_type} using Dropout {config.model.drop_rate}")
        model = models.resnet34(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(config.model.drop_rate),
            nn.Linear(model.fc.in_features, 64),
            nn.ReLU(True),
            nn.Dropout(config.model.drop_rate),
            nn.Linear(64, num_classes),
        )
    else:
        raise Exception('model type is not supported:', model_type)

    # if model_type == 'resnet34':
    #     print("ResNet34")
    #     model = models.resnet34(pretrained=True)
    #     model.fc = nn.Sequential(
    #         nn.Dropout(0.6),
    #         nn.Linear(model.fc.in_features, num_classes)
    #     )
    # elif model_type == 'resnet18':
    #     print("ResNet18")
    #     model = models.resnet18(pretrained=True)
    #     model.fc = nn.Sequential(
    #         nn.Dropout(0.6),
    #         nn.Linear(model.fc.in_features, num_classes)
    #     )
    # else:
    #     raise Exception('model type is not supported:', model_type)


    model.to('cuda')
    return model
