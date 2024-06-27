from torchvision.models import shufflenet_v2_x0_5
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, n_output_neurons, n_classes):
        super(CustomModel, self).__init__()
        self.shufflenet = shufflenet_v2_x0_5(pretrained=False)
        self.shufflenet.fc = nn.Identity()
        self.fc_bbox = nn.Linear(1024, n_output_neurons)
        self.fc_class = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.shufflenet(x)
        bbox_output = self.fc_bbox(x)
        class_output = self.fc_class(x)
        return {'bbox': bbox_output, 'class': class_output}