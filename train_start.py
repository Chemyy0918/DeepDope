import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.cuda.set_device(2)
torch.cuda.empty_cache()

# 1. 数据处理

def load_data(file_path):
    numpy_data=np.load(file_path)
    data = torch.from_numpy(numpy_data)  # 假设数据是Tensor格式
    X, y1, y2 = data[:, :8], data[:, 8], data[:, 9]
    X[:, 7] = X[:, 7].int()-1
    return X, y1, y2

def prepare_dataloader(X, y1, y2, batch_size):
    dataset = TensorDataset(X, y1, y2)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    return (DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=32,pin_memory=True,persistent_workers=True),
            DataLoader(val_set, batch_size=batch_size,num_workers=32,pin_memory=True,persistent_workers=True),
            DataLoader(test_set, batch_size=batch_size,num_workers=32,pin_memory=True,persistent_workers=True))

# 2. 定义模型

class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(250, embed_dim)
        
        # 公共特征提取层
        self.fc_common = nn.Sequential(
            nn.Linear(input_dim - 1 + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 双预测分支
        self.fc_branch1 = self._make_branch(hidden_dim)
        self.fc_branch2 = self._make_branch(hidden_dim)

        # 初始化权重
        self._init_weights()

    def _make_branch(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9)
        )

    def _init_weights(self):
        """ 何凯明初始化专项优化 """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 针对 SiLU 的推荐初始化配置
                nn.init.kaiming_normal_(module.weight, mode='fan_in',nonlinearity='relu')
                
                # 偏置项初始化为小常数
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

                # 对最后一层输出进行特殊初始化
                if module == self.fc_branch1[-1] or module == self.fc_branch2[-1]:
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        x[:, 7] = x[:, 7].int()
        embed = self.embedding(x[:, 7].long())
        x = torch.cat([x[:, :7], embed], dim=1)
        x = self.fc_common(x)
        out1 = self.fc_branch1(x)
        out2 = self.fc_branch2(x)
        return out1, out2

# 3. 负对数似然损失

def nll_loss(outputs, targets):
    log_weights = F.log_softmax(outputs[:, :3] - torch.max(outputs[:, :3], dim=1, keepdim=True)[0], dim=1)
    means = outputs[:, 3:6]
    log_stds = outputs[:, 6:9]

    diff = targets.unsqueeze(1).expand(-1, 3) - means
    prec = torch.exp(-2 * log_stds)
    log_likelihood = (-0.5 * (diff ** 2) * prec - log_stds - 0.5 * torch.log(2 * torch.tensor(torch.pi)))
    
    log_mixture = torch.logsumexp(log_weights + log_likelihood, dim=1)
    
    return -log_mixture.mean()


# 4. 训练函数

def train(model, train_loader, val_loader, device, epochs, lr):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2,min_lr=1e-6)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    loss_info=[]
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        batch_idx = 0
        for X_batch, y1_batch, y2_batch in train_loader:
            X_batch, y1_batch, y2_batch = X_batch.to(device,non_blocking=True), y1_batch.to(device,non_blocking=True), y2_batch.to(device,non_blocking=True)
            optimizer.zero_grad()
            out1, out2 = model(X_batch)
            loss = nll_loss(out1, y1_batch) + nll_loss(out2, y2_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 ==0:
                print(f"Train Loop - Batch {batch_idx}/{len(train_loader)} completed - Loss: {loss.item():.4f}")
            batch_idx=batch_idx+1
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        batch_idx = 0
        with torch.no_grad():
            for X_batch, y1_batch, y2_batch in val_loader:
                X_batch, y1_batch, y2_batch = X_batch.to(device,non_blocking=True), y1_batch.to(device,non_blocking=True), y2_batch.to(device,non_blocking=True)
                out1, out2 = model(X_batch)
                loss = nll_loss(out1, y1_batch) + nll_loss(out2, y2_batch)
                val_loss += loss.item()
                if batch_idx % 100 ==0:
                    print(f"Valid Loop - Batch {batch_idx}/{len(val_loader)} completed - Loss: {loss.item():.4f}")
                batch_idx=batch_idx+1
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        elapsed_time = time.time() - start_time
        print(f" ---- Epoch {epoch+1}/{epochs} - Time: {elapsed_time:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - lr: {optimizer.param_groups[0]['lr']}")
        loss_info.append([train_loss,val_loss])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'scheduler_state_dict': scheduler.state_dict()},os.path.join(save_dir, "best_model.pth"))
    np.savetxt(os.path.join(save_dir, "loss_info.txt"),np.array(loss_info).reshape(-1,2),fmt="%.6f")
    return save_dir
# 5. 测试函数

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y1_batch, y2_batch in test_loader:
            X_batch, y1_batch, y2_batch = X_batch.to(device,non_blocking=True), y1_batch.to(device,non_blocking=True), y2_batch.to(device,non_blocking=True)
            out1, out2 = model(X_batch)
            test_loss += (nll_loss(out1, y1_batch) + nll_loss(out2, y2_batch)).item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

# 6. 运行流程

def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    X, y1, y2 = load_data("Dataset/processed_data.npy")  # 请替换为你的数据文件
    train_loader, val_loader, test_loader = prepare_dataloader(X, y1, y2,batch_size=65536)
    model = MLP(input_dim=8, embed_dim=32, hidden_dim=2048)
    save_dir=train(model, train_loader, val_loader, device,epochs=36,lr=5e-4)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"),weights_only=True))
    test(model, test_loader, device)
    
if __name__ == "__main__":
    main()


