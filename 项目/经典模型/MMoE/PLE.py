import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from net import PLE
from torch.utils.data import DataLoader
from Dataset import censusData,load_and_process


train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = load_and_process()
batch_size=1024
train_loader=DataLoader(censusData(train_data,train_label),batch_size=batch_size,shuffle=True)
val_loader=DataLoader(censusData(validation_data,validation_label),batch_size=batch_size,shuffle=True)
test_loader=DataLoader(censusData(test_data,test_label),batch_size=batch_size,shuffle=True)


expert_size=16
tower_size=8
lr=1e-3
shared_num=2
task_num=2
exp_per_task=3
n_epochs=80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=PLE(input_size=train_data.shape[1],task_num=2,experts_per_task=exp_per_task,shared_experts=shared_num,expert_size=expert_size,num_layers=2)
model.to(device)

loss_fn=torch.nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

def test(loader,model):
    t1_pred,t2_pred,t1_tag,t2_tag=[],[],[],[]
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            yhat=model(x)
            y1,y2=y[:,0],y[:,1]
            yhat_1, yhat_2 = torch.sigmoid(yhat[0]), torch.sigmoid(yhat[1])

            t1_tag.append(y1.cpu().numpy())
            t2_tag.append(y2.cpu().numpy())

            t1_pred.append(yhat_1.detach().cpu().numpy())
            t2_pred.append(yhat_2.detach().cpu().numpy())
    t1_tag=np.concatenate(t1_tag)
    t2_tag=np.concatenate(t2_tag)

    t1_pred=np.concatenate(t1_pred)
    t2_pred=np.concatenate(t2_pred)

    auc_1=roc_auc_score(t1_tag,t1_pred)
    auc_2=roc_auc_score(t2_tag,t2_pred)
    return auc_1,auc_2

losses=[]
val_loss=[]

for epoch in tqdm(range(1,n_epochs+1)):
    model.train()
    epoch_loss=[]

    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        y_hat=model(x)

        y1,y2=y[:,0],y[:,1] # 真实标签
        y_1,y_2=y_hat[0],y_hat[1] # 预测输出

        loss1,loss2=loss_fn(y_1,y1.view(-1,1)),loss_fn(y_2,y2.view(-1,1))
        loss=loss1+loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
    losses.append(np.mean(epoch_loss))

    auc1,auc2=test(val_loader,model)
    if epoch==1 or epoch%10==0:print(f'epoch: {epoch}, train loss: {np.mean(epoch_loss)} val task1 auc: {auc1:.5f}, val task2 auc: {auc2:.3f}')

auc1,auc2=test(test_loader,model)
print(f'test auc1: {auc1:.3f}, test auc2: {auc2:.3f}')

# test auc1: 0.949, test auc2: 0.994