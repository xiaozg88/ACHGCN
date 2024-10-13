import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 加载 Cora 数据集
#dataset = Planetoid(root='/tmp/Cora', name='Cora')

dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
#dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
data = dataset[0]
class HGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, initial_curvature=-1.0):
        super(HGCNLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.linear = nn.Linear(in_channels, out_channels)
        self.curvature = nn.Parameter(torch.Tensor([initial_curvature]))  # 自适应曲率参数
        self.attention_linear = nn.Linear(2 * out_channels, 1)  # 注意力参数

    def forward(self, x, edge_index):
        # 添加自环以确保图卷积层包含节点自身的信息
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x)
        # 传播特征
        return self.propagate(edge_index, x=x, curvature=-torch.abs(self.curvature))

    def message(self, x_i, x_j, edge_index, curvature):
        # 拼接源节点和目标节点的特征
        x_concat = torch.cat([x_i, x_j], dim=1)
        alpha = self.attention_linear(x_concat)  # 计算注意力权重
        alpha = F.leaky_relu(alpha)
        alpha = F.softmax(alpha, dim=1)
        return alpha * x_j * curvature  # 使用自适应曲率和注意力机制

    def update(self, aggr_out):
        # 更新节点特征
        return aggr_out

    def get_curvature(self):
        # 获取当前曲率值
        return -torch.abs(self.curvature).item()

class HGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, initial_curvature=-1.0):
        super(HGCN, self).__init__()
        self.conv1 = HGCNLayer(in_channels, hidden_channels, initial_curvature)
        self.conv2 = HGCNLayer(hidden_channels, out_channels, initial_curvature)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def get_curvatures(self):
        # 获取所有层的曲率值
        return [self.conv1.get_curvature(), self.conv2.get_curvature()]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HGCN(dataset.num_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_losses = []
train_accuracies = []
val_accuracies = []
curvatures = []

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # 计算训练准确率
    pred = out.argmax(dim=1)
    correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    acc = int(correct) / int(data.train_mask.sum())
    return loss.item(), acc

def validate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc



# 训练模型并记录训练和验证准确率
for epoch in range(200):
    train_loss, train_acc = train()
    val_acc = validate()

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    curvatures.append(model.get_curvatures())

    print(f'Epoch {epoch + 1:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# 绘制训练和验证准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize='large')
plt.title('Training and Validation Accuracy based on the Pubmed dataset', fontsize=14)
plt.savefig('Train and Validation Accuracy based on the Pubmed dataset.jpg')  # 保存图像
plt.show()

# 绘制曲率变化曲线
curvatures = torch.tensor(curvatures)
plt.figure(figsize=(10, 5))
for i in range(curvatures.shape[1]):
    plt.plot(curvatures[:, i].numpy(), label=f'Curvature Layer {i + 1}')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Curvature', fontsize=14)
plt.legend(fontsize='large')
plt.title('Curvature Changes Over Epochs based on the Pubmed dataset', fontsize=14)
plt.show()

# 定义类别名称
#class_names = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
#               'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']

# 定义类别名称
class_names = ["Diabetes Mellitus", "Cancer", "Cardiovascular Diseases"]
# 定义类别名称
#class_names = ['Agents', 'AI (Artificial Intelligence)', 'DB (Database)',
#               'IR (Information Retrieval)', 'ML (Machine Learning)', 'HCI (Human-Computer Interaction)']


# 定义 t-SNE 可视化函数
def plot_tsne(embeddings, labels, title='t-SNE'):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 5))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label = class_names[label], s=10)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize='large')
    plt.title(title, fontsize=14)
    plt.savefig(title+'.jpg')  # 保存图像
    plt.show()

# 获取节点嵌入并绘制 t-SNE 图
model.eval()
embeddings = model.get_embeddings(data.x, data.edge_index).detach().cpu().numpy()
labels = data.y.cpu().numpy()
plot_tsne(embeddings, labels, title='t-SNE of Node Embeddings based on the Pubmed dataset')
