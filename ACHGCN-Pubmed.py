import torch

import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
# 加载数据集

# 加载 Cora 数据集
dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
data = dataset[0]
# 定义模型
class GAT_HGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT_HGCN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        return F.elu(x)

# 初始化模型、优化器和损失函数
model = GAT_HGCN(dataset.num_features, 8, dataset.num_classes, heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    # 计算聚类损失
    embeddings = model.get_embeddings(data.x, data.edge_index).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=dataset.num_classes, random_state=0).fit(embeddings)
    cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=data.y.device)
    cluster_loss = F.nll_loss(out[data.train_mask], cluster_labels[data.train_mask])

    # 总损失
    total_loss = loss + cluster_loss
    total_loss.backward()
    optimizer.step()

    # 计算训练准确率
    pred = out.argmax(dim=1)
    correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    acc = int(correct) / int(data.train_mask.sum())
    return total_loss.item(), acc

# 验证函数
def validate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc

# 记录训练和验证准确率
train_losses = []
train_accuracies = []
val_accuracies = []

# 训练模型并记录训练和验证准确率
for epoch in range(200):
    train_loss, train_acc = train()
    val_acc = validate()

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f'Epoch {epoch + 1:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')



# 绘制训练和验证准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Train and Validation Accuracy based on the Pubmed dataset', fontsize=14)
plt.legend(fontsize='large')
plt.savefig('Training and Validation Accuracy based on the Pubmed dataset.jpg')  # 保存图像
plt.show()

# 定义类别名称
class_names = ["Diabetes Mellitus", "Cancer", "Cardiovascular Diseases"]

# 定义 t-SNE 可视化函数
def plot_tsne(embeddings, labels, title='t-SNE'):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 5))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=class_names[label], s=10)
    plt.legend(fontsize='large')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(title, fontsize=14)
    plt.savefig(title + '.jpg')  # 保存图像
    plt.show()

# 获取节点嵌入并绘制 t-SNE 图
model.eval()
embeddings = model.get_embeddings(data.x, data.edge_index).detach().cpu().numpy()
labels = data.y.cpu().numpy()
plot_tsne(embeddings, labels, title='t-SNE of Node Embeddings based on the Pubmed dataset')
