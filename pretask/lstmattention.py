import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
class LSTMAttention(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=1, bidirectional=False,
                            batch_first=True)
        self.attention = nn.Linear(embed_dim, 1)
        self.seq_len = seq_len

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        att = F.softmax(self.attention(lstm_out), dim=1)
        att_out = torch.bmm(att.permute(0, 2, 1), lstm_out).squeeze()
        return att_out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViT_LSTM(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, seq_len, hidden_dim, num_heads, num_layers,
                 dropout, classification):
        super(ViT_LSTM, self).__init__()

        # Image and patch sizes
        self.image_size = image_size
        self.patch_size = patch_size
        self.classification = classification
        # Number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels=4, out_channels=embed_dim, kernel_size=patch_size,
                                         stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        # Position embedding
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Encoder layers
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                LSTMAttention(embed_dim, seq_len),
                nn.LayerNorm(embed_dim),
                nn.MultiheadAttention(embed_dim, num_heads),
                nn.LayerNorm(embed_dim),
                MLP(embed_dim, hidden_dim, embed_dim, dropout),
                nn.LayerNorm(embed_dim)
            ]))

        # Classifier

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Image to patches
       # x = x.reshape(-1, 22, 114, 9)
        x = x.reshape (-1,4,32,32)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        # Add position embedding
        #print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
       # print(cls_tokens.shape)

        #x = torch.cat([self.position_embedding[:, :self.num_patches], x], dim=1)
        x = torch.cat([cls_tokens, x], dim=1)
       # print(x.shape)
        # Transformer encoder layers
        for [lstm_att, norm1, attention, norm2, mlp, norm3] in self.layers:
            x_lstm_att = lstm_att(x).unsqueeze(1)
            x = x + x_lstm_att
            x = norm1(x)
           # x_att, _ = attention(x, x, x)
            #x = x + x_att
            x = norm2(x)
            x_mlp = mlp(x)
            x = x + x_mlp
            x = norm3(x)
        #print(x.shape)
       # x = x.mean(dim=1)

        #x = self.classifier(x)
        return x

class MY_NET(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, seq_len, hidden_dim, num_heads, num_layers,
                 dropout, classification):
        super(MY_NET, self).__init__()
        self.ViT_LSTM = ViT_LSTM(image_size, patch_size, num_classes, embed_dim, seq_len, hidden_dim, num_heads, num_layers,
                 dropout, classification)
        self.classification = classification
        if self.classification == True:
            for param in self.ViT_LSTM.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(256, 384)
        self.bn3 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 192)
        self.bn4 = nn.BatchNorm1d(192)
        self.fc3 = nn.Linear(192, 2)
        self.softmax = nn.Softmax(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # self.conv1 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x =self.ViT_LSTM(x)
        return x
        #if self.classification:

            # x = x.reshape(-1, 256)
            # x = x.reshape(-1, 16, 16)
            # x = self.conv1(x)
            # x = torch.relu(x)
            # x = torch.max_pool1d(x, kernel_size=2, stride=2)
            # x = torch.relu(self.conv2(x))
            # x = torch.max_pool1d(x, kernel_size=2, stride=2)
            # x = torch.relu(self.conv3(x))
            # x = torch.max_pool1d(x, kernel_size=2, stride=2)
            # x = x.view(x.size(0), -1)
            # x = torch.relu(self.fc1(x))
            # x = torch.relu(self.fc2(x))
            # x = self.fc3(x)

            # x = self.fc1(x)
            # x = self.bn3(x)
            # x = self.relu(x)
            # x = self.dropout(x)
            # x = self.fc2(x)
            # x = self.bn4(x)
            # x = self.relu(x)
            # x = self.dropout(x)
            # x = self.fc3(x)
