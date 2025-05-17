import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    
model=MLP(input_dim=8, embed_dim=32, hidden_dim=2048)

model_pth=input("输入你的模型目录")
model.load_state_dict(torch.load(model_pth+"/best_model.pth",weights_only=True))
model.to("cpu")
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
element_percent=input("输入材料元素占比 小数 按空格分开").split()
element_percent = list(map(float, element_percent))
element_percent=torch.tensor(element_percent)

element_percent=torch.sigmoid(element_percent)
kpath = torch.tensor([0.0000,0.0420,0.0840,0.1260,0.1680,0.2100,0.2520,0.2941,0.3361,0.3781,0.4201,0.4621,0.5041,0.5461,0.5881,0.6301,0.6721,0.7141,0.7561,0.7981,0.8402,0.8822,0.9242,0.9662,1.0082,1.0502,1.0922,1.1342,1.1762,1.2182,1.2602,1.3022,1.3443,1.3863,1.4283,1.4703,1.4703,1.5127,1.5552,1.5976,1.6400,1.6825,1.7249,1.7674,1.8098,1.8523,1.8947,1.9372,1.9796,2.0220,2.0645,2.1069,2.1494,2.1918,2.2343,2.2767,2.3191,2.3191,2.3605,2.4020,2.4434,2.4848,2.5262,2.5676,2.6090,2.6504,2.6918,2.7332,2.7746,2.8160,2.8574,2.8989,2.9403,2.9817,3.0231,3.0645,3.1059,3.1473,3.1887,3.2301,3.2715,3.3129,3.3543,3.3957,3.4372,3.4786,3.5200,3.5614,3.6028,3.6442,3.6856,3.7270,3.7684,3.8098,3.8512,3.8926,3.9341,3.9755,4.0169],dtype=torch.float32)
kpath=kpath/max(kpath)
band_index = torch.arange(0, 250,dtype=torch.float32)  # 生成 1 到 250
tmp_list=[]
for k in kpath:
    for b in band_index:
        tmp_tensor = torch.cat([element_percent, torch.tensor([k, b])])
        tmp_list.append(tmp_tensor)

X = torch.stack(tmp_list)
with torch.no_grad():
    yE, yW = model(X)
softmax_outputE = F.softmax(yE[:, :3]-torch.max(yE[:, :3],dim=1,keepdim=True)[0],dim=1)
softmax_outputW = F.softmax(yW[:, :3]-torch.max(yE[:, :3],dim=1,keepdim=True)[0],dim=1)
E = softmax_outputE[:, 0] * yE[:, 3] + softmax_outputE[:, 1] * yE[:, 4] + softmax_outputE[:, 2] * yE[:, 5]
W = softmax_outputW[:, 0] * yW[:, 3] + softmax_outputW[:, 1] * yW[:, 4] + softmax_outputW[:, 2] * yW[:, 5]
E=20*torch.atanh(E)
EBS_pred=torch.cat((X[:,6].reshape(-1,1),E.reshape(-1,1),W.reshape(-1,1)),dim=1)
EBS_pred=EBS_pred.numpy()
np.savetxt("EBS_pred.dat",EBS_pred)
