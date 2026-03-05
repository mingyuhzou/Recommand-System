import torch
import torch.nn as nn
import torch.nn.functional as F
from pyarrow.types import is_large_string


class SinglePLELayer(nn.Module):
    def __init__(self,input_size,task_num,exper_per_task,shared_experts,experts_size,if_last):
        super(SinglePLELayer, self).__init__()
        self.input_size = input_size
        self.task_num = task_num # 任务数
        self.exper_per_task = exper_per_task # 任务特定专家有多少个自主专家
        self.shared_experts = shared_experts # 共享专家有多少个子专家
        self.experts_size = experts_size # PLE中间层的宽度
        self.if_last = if_last # 是否是最后一层

        self.task_experts=nn.ModuleList()
        for _ in range(task_num):
            experts=nn.ModuleList([nn.Linear(input_size,experts_size) for _ in range(exper_per_task)])
            self.task_experts.append(experts)

        self.shared_experts=nn.ModuleList([nn.Linear(input_size,experts_size) for _ in range(shared_experts)])

        self.task_gates=nn.ModuleList()
        gate_out_dim=exper_per_task+shared_experts
        for _ in range(task_num):
            self.task_gates.append(nn.Linear(input_size,gate_out_dim))

        if not if_last:
            shared_gate_out_dim=task_num*exper_per_task+shared_experts
            self.shared_gate=nn.Linear(
                input_size,shared_gate_out_dim
            )

        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self,inputs):
        """
        inputs: list of Tensor
            len=task_num+1
            inputs[i] -> task i feature, shape[B,D]
            inputs[-1] -> shared feature, shape[B,D]
        """

        '''
        task_expert_out=[
            [Tensor(B,D),...,Tensor(B,D)], task 0
            [Tensor(B,D),...,Tensor(B,D)], task 1
        ]
        '''
        task_expert_out=[]
        for i in range(self.task_num):
            outs=[
                F.relu(expert(inputs[i])) for expert in self.task_experts[i]
            ]
            task_expert_out.append(outs)

        '''
        shared_expert_out=[Tensor(B,D),...,Tensor(B,D)]
        '''
        shared_expert_out=[F.relu(expert(inputs[-1])) for expert in self.shared_experts]

        outputs=[]

        for i in range(self.task_num):
            experts=task_expert_out[i]+shared_expert_out # [Tensor(B,D),...,Tensor(B,D)] 共exper_per_task+shared_experts
            experts=torch.stack(experts,dim=1) # [B,exper_per_task+shared+num,D]

            gate=self.task_gates[i](inputs[i]) # [B,gate_out_dim]
            gate=F.softmax(gate,dim=1).unsqueeze(-1) # [B, gate_out_dim, 1] gate_out_dim=exper_per_task+shared_experts

            # 门控加权
            out=torch.sum(experts*gate,dim=1) # [B,D]
            outputs.append(out)

        if not self.if_last:
            all_experts=[]
            for i in range(self.task_num):
                all_experts.extend(task_expert_out[i])
            all_experts.extend(shared_expert_out) # [Tensor(B,D),...,Tensor(B,D)] task*exper_per_task+shared_experts

            experts=torch.stack(all_experts,dim=1) # [B,task*exper_per_task+shared_num,D]

            gate=self.shared_gate(inputs[-1])
            gate=F.softmax(gate,dim=1).unsqueeze(-1)

            shared_out=torch.sum(experts*gate,dim=1)
            outputs.append(shared_out) # 加在最后，满足inputs[-1]是shared feature
        return outputs


class PLE(nn.Module):
    def __init__(self,input_size,task_num,experts_per_task,shared_experts,expert_size,num_layers):
        super(PLE, self).__init__()

        self.layers=nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SinglePLELayer(input_size if i==0 else expert_size,
                task_num,experts_per_task,shared_experts,expert_size,if_last=(i==num_layers-1))
            )

        self.towers=nn.ModuleList([
            nn.Linear(expert_size,1) for _ in range(task_num)
        ])
    def forward(self,X):
        inputs=[X for _ in range(self.layers[0].task_num+1)]

        for layer in self.layers:
            inputs=layer(inputs)
        outputs=[]
        for t in range(len(self.towers)):
            outputs.append(self.towers[t](inputs[t]))
        return outputs