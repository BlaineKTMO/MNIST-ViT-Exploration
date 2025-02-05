#!/usr/bin/env python3

from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gc

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.Wq(Q))
        K = self.split_heads(self.Wk(K))
        V = self.split_heads(self.Wv(V))
        
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        output = self.Wo(self.combine_heads(attn_output))
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.ff2(self.relu(self.ff1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, seq_length, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        self.pixel_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        self.encoder_layers = nn.ModuleList([Encoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)
        x = self.pixel_proj(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        attn_weights_list = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            if return_attn:
                attn_weights_list.append(attn_weights)
        
        x = x.mean(dim=1)
        x = self.fc(x)

        if return_attn:
            return x, attn_weights_list
        else:
            return x

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / len(train_loader))
    
    return running_loss / len(train_loader)

def main():
    sns.set_theme(style='whitegrid')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model setup
    model = VisionTransformer(
        d_model=64, num_heads=8, d_ff=256,
        dropout=0.1, seq_length=28*28,
        num_layers=4, num_classes=10
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
  
    # Training loop
    num_epochs = 1
    train_losses = []
    test_losses = []
    test_accuracies = []
    all_preds = []
    all_true = []

    sample_images = []
    sample_true = []
    sample_preds = []
    all_attentions = []
        

    for epoch in range(num_epochs):
        # Train with progress bar
        model.load_state_dict(torch.load('/home/blaine/neural_networks/model.pth'))
        model.train()
        epoch_train_loss = 0
        train_bar = tqdm(train_loader, 
                        desc=f"🚀 Epoch {epoch+1}/{num_epochs} [Training]", 
                        bar_format="{l_bar}{bar:20}{r_bar}",
                        colour="#00ff00")
        
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            torch.save(model.state_dict(), 'model.pth')
            
            epoch_train_loss += loss.item()
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_train_loss/(batch_idx+1):.4f}"
            })

        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Test with progress bar
        model.eval()
        test_loss = 0
        correct = 0
        test_bar = tqdm(test_loader, 
                    desc=f"🔍 Epoch {epoch+1}/{num_epochs} [Testing]", 
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    colour="#ffaa00")
        

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_bar):
                data, target = data.to(device), target.to(device)
                output, attn_weights = model(data, return_attn=True)
                
                # Calculate metrics
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                # Update progress bar
                test_bar.set_postfix({
                    'loss': f"{test_loss/(batch_idx+1):.4f}",
                    'acc': f"{correct/((batch_idx+1)*data.size(0)):.2%}"
                })
                
                # Collect samples and predictions
                if batch_idx == 0:
                    sample_images = data[:5].cpu()
                    sample_true = target[:5].cpu()
                    sample_preds = pred[:5].cpu()
                
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(target.cpu().numpy())

                all_attentions.append(attn_weights)
        
        # Calculate final metrics
        test_loss /= len(test_loader)
        test_acc = correct / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Print epoch summary
        tqdm.write(f"\n🌈 Epoch {epoch+1} Summary:")
        tqdm.write(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}\n")
        sleep(0.1)  # Pause for clean output

    # Visualization
    plt.figure(figsize=(18, 10))
    
    # Sample predictions
    plt.subplot(2, 3, 1)
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(sample_images[i].squeeze(), cmap='gray')
        plt.title(f'True: {sample_true[i]}\nPred: {sample_preds[i]}', 
                 color='green' if sample_true[i] == sample_preds[i] else 'red')
        plt.axis('off')
    
    # Attention map (dummy example - replace with actual implementation)
    plt.subplot(2, 5, 6)
    attention_map = plot_attention_maps(sample_images, sample_true, sample_preds, all_attentions)  # Replace with actual attention map
    plt.imshow(attention_map, cmap='hot')
    plt.title('Attention Map')
    plt.axis('off')
    
    # Training curves
    plt.subplot(2, 3, 2)
    sns.lineplot(x=range(1, num_epochs+1), y=train_losses, marker='o', label='Train Loss')
    sns.lineplot(x=range(1, num_epochs+1), y=test_losses, marker='o', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training/Test Loss Curve')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    sns.lineplot(x=range(1, num_epochs+1), y=test_accuracies, marker='o', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Curve')
    
    # Confusion matrix
    plt.subplot(2, 3, 6)
    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()

def plot_attention_maps(images, true_labels, pred_labels, attn_weights):
    plt.figure(figsize=(15, 10))
    
    # Select first sample from batch
    img = images[0].squeeze()
    true = true_labels[0].item()
    pred = pred_labels[0].item()
    
    # Select attention from first layer and average across heads
    layer_attn = attn_weights[0][0]  # First layer, first batch
    attn_map = layer_attn.mean(dim=0)  # Average across heads
    attn_map = attn_map.mean(dim=0)  # Average across query positions
    attn_map = attn_map.view(28, 28).cpu().numpy()
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'True: {true}, Pred: {pred}')
    
    # Plot attention map
    plt.subplot(1, 2, 2)
    plt.imshow(attn_map, cmap='hot')
    plt.title('Attention Map')
    plt.colorbar()
    
    plt.show()

if __name__ == "__main__":
    main()