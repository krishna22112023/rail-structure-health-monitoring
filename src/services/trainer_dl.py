import glob
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import confusion_matrix

from src.services.modules import ANN, CNN  

class TrainerDL:
    def __init__(self, model_name, model_params, data_dir):
        self.model_params = model_params
        self.model = self.get_model(model_name, model_params)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.create_dataloaders(data_dir, model_params["batch_size"])
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(self.model, len(self.train_dataloader), init_lr=model_params["init_lr"], weight_decay=model_params["weight_decay"])

    def create_dataloaders(self, data_dir, batch_size):

        def find_file(prefix):
            """Find the first file in the directory that matches the given prefix."""
            files = glob.glob(os.path.join(data_dir, f"{prefix}*.pt"))
            if not files:
                raise FileNotFoundError(f"No file found with prefix {prefix} in {data_dir}")
            return files[0]  # Return the first matching file

        # Locate files
        X_train_path = find_file("X_train")
        y_train_path = find_file("y_train")
        X_val_path = find_file("X_val")
        y_val_path = find_file("y_val")
        X_test_path = find_file("X_test")
        y_test_path = find_file("y_test")

        # Load data using torch
        X_train = torch.load(X_train_path)
        y_train = torch.load(y_train_path)
        X_val = torch.load(X_val_path)
        y_val = torch.load(y_val_path)
        X_test = torch.load(X_test_path)
        y_test = torch.load(y_test_path)

        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader

    def get_model(self, model_name, model_params):
        if model_name == "ann":
            model = ANN(model_params["hidden_dim"], model_params["num_classes"], model_params["num_layers"], model_params["dropout"])
        elif model_name == "cnn":
            model = CNN(model_params["input_channels"], model_params["num_cnn_layers"], model_params["hidden_dim"], model_params["num_classes"],model_params["num_layers"], model_params["dropout"])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model

    def get_optimizer_and_scheduler(self, model, total_steps, init_lr=0.01, weight_decay=0.01):
        optimizer = AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        return optimizer, scheduler
    
    def compute_confusion_matrix(self, y_true, y_pred, labels=None):
        
        # Convert tensors to numpy arrays if necessary
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm

    def train_model(self):
        num_epochs = self.model_params["num_epochs"]
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.model.to(device)
        metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc": 0, "test_avg_latency":0}
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            for x, y in self.train_dataloader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()  # Zero out the gradients
                outputs = self.model(x)     # Forward pass
                loss = loss_fn(outputs, y)
                loss.backward()        # Backpropagation
                self.optimizer.step()       # Update parameters
                
                # Accumulate training loss and accuracy
                train_loss += loss.item() * x.size(0)
                _, preds = outputs.max(1)
                train_correct += (preds == y).sum().item()
                train_samples += x.size(0)
            
            avg_train_loss = train_loss / train_samples
            train_accuracy = train_correct / train_samples
            
            metrics["train_loss"].append(avg_train_loss)
            metrics["train_acc"].append(train_accuracy)
            
            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for x, y in self.val_dataloader:
                    x, y = x.to(device), y.to(device)
                    outputs = self.model(x)
                    loss = loss_fn(outputs, y)
                    val_loss += loss.item() * x.size(0)
                    _, preds = outputs.max(1)
                    val_correct += (preds == y).sum().item()
                    val_samples += x.size(0)
                    
            avg_val_loss = val_loss / val_samples
            val_accuracy = val_correct / val_samples

            metrics["val_loss"].append(avg_val_loss)
            metrics["val_acc"].append(val_accuracy)
            
            # Step the scheduler after each epoch
            self.scheduler.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
        # test data predictions
        self.model.eval()
        all_preds = []
        all_labels = []
        total_time = 0
        test_correct = 0
        test_samples = 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x, y = x.to(device), y.to(device)
                # Synchronize GPU if applicable
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                outputs = self.model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                _, preds = outputs.max(1)
                test_correct += (preds == y).sum().item()
                test_samples += x.size(0)
                
                #compute total time
                batch_time = end_time - start_time
                total_time += batch_time

                # Collect predictions and true labels
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
                
        # Compute confusion matrix
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        cm = self.compute_confusion_matrix(all_labels, all_preds)

        test_accuracy = test_correct / test_samples
        test_avg_latency = total_time / test_samples
        print(f"Test Acc: {test_accuracy:.4f}, Test Latency: {test_avg_latency:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        metrics["test_acc"] = test_accuracy
        metrics["test_avg_latency"] = test_avg_latency
            
        return metrics, self.model, cm 