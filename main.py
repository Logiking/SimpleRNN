import time
import datetime
import argparse
import torch
import torch.nn as nn
from dataset import load_mnist
from model import RNN

parser = argparse.ArgumentParser(prog='main',
                                 description="parameters for training model",
                                 )
parser.add_argument('-lr', '--learning-rate', help="specify learning rate", default=0.01, type=float)
parser.add_argument('-bs', '--batch-size', help="specify batch size", default=64, type=int)
parser.add_argument('-hid_dim', '--hidden_dim', help="specify hiddden dim", default=128, type=int)
parser.add_argument('-optim', '--optimizer', help="specify optimizer", choices=['adam', 'sgd'], default='sgd', type=str)
parser.add_argument('-ep', '--epochs', help="specify epochs", default=50, type=int)

args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def accuracy_check(model, loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for img, target in loader:
            img, target = img.to(device), target.to(device)
            x = model(img)
            _, pred = torch.max(x, dim=-1)
            correct += (pred == target).sum().item()
            total += img.size(0)
        return correct / total

def main():
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    train_loader, test_loader = load_mnist(batch_size=batch_size)
    model = RNN(28, 10, hidden_dim)
    model = model.to(device)
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    acc = accuracy_check(model, test_loader, device)
    print(f"Before Training, The Accuracy of Test Loader is {acc}.")
    
    Loss = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        LOSS = 0
        total = 0
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            x = model(img)
            loss = Loss(x, target)
            
            LOSS += loss.item()
            total += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.eval()
        Accuracy = accuracy_check(model, test_loader, device) * 100
        c_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Epoch: [{epoch:>2}/{args.epochs}] | Train Loss: {(LOSS / total):<7.4f} | "
              f"Accuracy: {Accuracy:<6.2f}% | LR: {optimizer.param_groups[0]['lr']:<5.5f} | "
              f"Time: {(time.time() - start_time):<6.2f}s | Now: {c_time}.")



if __name__ == "__main__":
    main()
