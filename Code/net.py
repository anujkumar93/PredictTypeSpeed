import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.hidden1 = nn.Linear(n_feature, 512)   # hidden layer
        self.bn1 = nn.BatchNorm1d(512)

        self.hidden2 = nn.Linear(512, 512)  # hidden layer
        self.bn2 = nn.BatchNorm1d(512)

        self.hidden3 = nn.Linear(512, 512)  # hidden layer
        self.bn3 = nn.BatchNorm1d(512)

        self.hidden4 = nn.Linear(512, 512)  # hidden layer
        self.bn4 = nn.BatchNorm1d(512)

        self.hidden5 = nn.Linear(512, 512)  # hidden layer
        self.bn5 = nn.BatchNorm1d(512)

        self.hidden6 = nn.Linear(512, 512)  # hidden layer
        self.bn6 = nn.BatchNorm1d(512)

        self.hidden7 = nn.Linear(512, 256)  # hidden layer
        self.predict = nn.Linear(256, 1)   # output layer

    def forward(self, batch):
        output_layer1 = self.bn1(self.relu(self.hidden1(batch)))
        output_layer2 = self.bn2(self.relu(self.hidden2(output_layer1)))
        output_layer3 = self.bn3(self.relu(self.hidden3(output_layer2)))
        output_layer4 = self.bn4(self.relu(self.hidden4(output_layer3)))
        output_layer5 = self.bn5(self.relu(self.hidden5(output_layer4)))
        output_layer6 = self.bn6(self.relu(self.hidden6(output_layer5)))

        output_layer7 = self.hidden7(output_layer6)
        output_preds = self.predict(output_layer7)
        return output_preds

