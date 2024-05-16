import torch
from data import DataModule
from model import BERTClass
import numpy as np
import gc

EPOCHS = 3
LEARNING_RATE = 1e-5
TOKENIZER = 'bert-base-uncased'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def train(model,epoch,training_loader):
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    model.train()
    for i,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if i%20==0:
            print("\r",end = '')
            print(f'Epoch {epoch+1}: {str(round(i/len(training_loader), 2))}, Loss:  {loss.item():.4f}',end='')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ids.detach()
        # mask.detach()
        # token_type_ids.detach()
        # targets.detach()
    print()

def validation(model,testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            predictions = torch.argmax(outputs,dim=1)

            fin_targets.extend(targets.cpu().numpy())
            fin_outputs.extend(predictions.cpu().numpy())
    correct = np.sum(np.array(fin_outputs) == np.array(fin_targets))
    accuracy = correct / len(fin_outputs)
    return accuracy

def main():
    print('start training')
    data_model = DataModule('bert-base-uncased')
    data_model.setup()
    training_loader = data_model.get_train_dataloader()
    testing_loader = data_model.get_test_dataloader()

    gc.collect()
    torch.cuda.empty_cache()
    model = BERTClass()
    model.to(device)
    for epoch in range(EPOCHS):
        train(model,epoch,training_loader)
    torch.save(model.state_dict(),"model_acc:.pt")
    acc = validation(model,testing_loader)
    print(f'test acc: {acc}')

if __name__ == "__main__":
    main()
    