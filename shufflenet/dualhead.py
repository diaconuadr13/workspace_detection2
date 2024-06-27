import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import shufflenet_v2_x0_5


class CustomModel(nn.Module):
    def __init__(self, n_output_neurons, n_classifier_neurons, n_bbox_neurons):
        super(CustomModel, self).__init__()
        self.base_model = shufflenet_v2_x0_5(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, n_output_neurons)
        self.classifier = nn.Linear(n_output_neurons, n_classifier_neurons)
        self.bbox_predictor = nn.Linear(n_output_neurons, n_bbox_neurons)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        class_predict = self.classifier(x)
        class_predict = self.sigmoid(class_predict)
        bbox_predict = self.bbox_predictor(x)
        bbox_predict = self.sigmoid(bbox_predict)  # Apply sigmoid to bounding box prediction
        return class_predict, bbox_predict