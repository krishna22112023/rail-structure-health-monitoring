from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import torch
import time
import os
import glob

class TrainerSklearn:

    def __init__(self, model_name, model_params, data_dir):
        model_params = self.process_params(model_params)
        self.model = self.get_model(model_name, model_params)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_data(data_dir)
        self.X_train, self.X_val, self.X_test = self.standardize_data(self.X_train, self.X_val, self.X_test)

    def load_data(self,data_dir):
        """Load train and validation features and labels from .npy files."""

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
        X_train = torch.load(X_train_path).cpu().numpy()
        y_train = torch.load(y_train_path).cpu().numpy()
        X_val = torch.load(X_val_path).cpu().numpy()
        y_val = torch.load(y_val_path).cpu().numpy()
        X_test = torch.load(X_test_path).cpu().numpy()
        y_test = torch.load(y_test_path).cpu().numpy()

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def process_params(self,params):
        """Convert string values to proper Python types."""
        for key, value in params.items():
            if isinstance(value, str):
                if value.lower() == "true":
                    params[key] = True
                elif value.lower() == "false":
                    params[key] = False
                elif value.lower() == "none":
                    params[key] = None
                elif value == "":  # Handle empty strings
                    params[key] = None
        return params

    def standardize_data(self,X_train, X_val, X_test):
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled

    def get_model(self,model_name, model_params):
        """Instantiate a scikit-learn model based on the model name and parameters."""
        model_name = model_name.lower()
        if model_name == "grad_boost":
            model = GradientBoostingClassifier(**model_params)
        elif model_name == "random_forest":
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model
    
    def compute_confusion_matrix(self, y_true, y_pred, labels=None):
        
        # Convert tensors to numpy arrays if necessary
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm

    def train_model(self):
        """
        Fit the model and compute training and validation loss (categorical cross entropy)
        as well as accuracy.
        """
        #fit the model
        self.model.fit(self.X_train, self.y_train)
        
        # For loss computation, we require probability estimates.
        y_train_proba = self.model.predict_proba(self.X_train)
        y_val_proba = self.model.predict_proba(self.X_val)
        
        # Compute categorical cross entropy loss (log_loss)
        train_loss = log_loss(self.y_train, y_train_proba)
        val_loss   = log_loss(self.y_val, y_val_proba)
        
        # Also compute accuracy for additional insight
        train_acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
        val_acc   = accuracy_score(self.y_val, self.model.predict(self.X_val))

        #compute accuracy on test data
        start_time = time.time()
        test_pred = self.model.predict(self.X_test)
        end_time = time.time()
        total_time = end_time - start_time
        avg_latency = total_time / len(self.X_test)
        test_acc = accuracy_score(self.y_test, test_pred)

        #compute confusion matrix
        cm = self.compute_confusion_matrix(self.y_test, test_pred)

        metrics = {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "test_acc": test_acc, "test_avg_latency": avg_latency}
        
        return metrics, self.model, cm

    