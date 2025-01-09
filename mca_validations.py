import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# Customized 3D CNN model to adapt to the input shape
class cnn_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, padding=(0, 1, 1))

        self.conv21 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, padding=0)

        self.fc_layer = nn.Linear(64*8*1*1, 2)
        self.dropout_layer = nn.Dropout(p=0.3)

    def forward(self, xb):
        h1 = self.conv11(xb)
        h1 = self.conv12(h1)
        h1 = self.pool1(h1)
        h1 = F.relu(h1)

        h2 = self.conv21(h1)
        h2 = self.conv22(h2)
        h2 = self.pool2(h2)
        h2 = F.relu(h2)

        # Before the fully connected layer, we need to flatten the output
        flatten = h2.view(-1, 64*8*1*1)
        out = self.fc_layer(flatten)
        return out


# The default validation type is 'arousal', change to validate others
# ['valence', 'arousal', 'dominance', 'liking']
validation_type = 'arousal'
validation_model = f'{validation_type}_model.pth'
validation_x_test = f'{validation_type}_x_test.pth'
validation_y_test = f'{validation_type}_y_test.pth'

validation_type = 'MCA_valence'
validation_model = f'{validation_type}_lyl.pth'
validation_x_test = f'{validation_type}_x_lyl.pth'
validation_y_test = f'{validation_type}_y_lyl.pth'

# Load model from the .pth file
model_state_dict = torch.load(f'models/{validation_model}', map_location=torch.device('cuda'))
# print(model_state_dict)
model = cnn_classifier()



new_state_dict = {}
for k, v in model_state_dict.items():
    if k.startswith('module.'):
        new_key = k[len('module.'):]
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.cuda()
# model.load_state_dict(model_state_dict)

# Load the x_test and y_test
x_test = torch.load(f'models/{validation_x_test}')
y_test = torch.load(f'models/{validation_y_test}')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Predict process
with torch.no_grad():
    outputs = model(x_test.view(-1, 1, 32, 5, 3).float().to(device))
    predicted = outputs.argmax(1)

# Convert to numpy array
predicted = predicted.cpu().numpy()
y_test = y_test.cpu().numpy()
# predicted = predicted.numpy()
# y_test = y_test.numpy()

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
conf_matrix = confusion_matrix(y_test, predicted)
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['True: Low Valence', 'True: High Valence'],
                              columns=['Predict: Low Valence', 'Predict: High Valence'])

print(f"Validation Model: {validation_type}")
print(f"Model Accuracy: {accuracy}, Model precision: {precision}, Model recall: {recall}, Model f1: {f1}")
print(conf_matrix_df)