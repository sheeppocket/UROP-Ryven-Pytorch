from ryven.node_env import *
import torch.nn as nn


class LossFunctionSelector(Node):
    """
    Selects and returns a loss function based on the specified type.
    Supports various loss functions for classification and regression tasks.
    """
    title = "Loss Function"
    init_inputs = [
        NodeInputType(label="loss_type"),
    ]
    init_outputs = [
        NodeOutputType(label="loss_function"),
    ]
    
    def update_event(self):
        loss_type = self.input(0)
        
        if loss_type is None:
            loss_type = "cross_entropy"
        
        loss_type = str(loss_type).lower()
        
        try:
            # Binary Classification
            if loss_type == "bce_logits" or loss_type == "bcewithlogitsloss":
                loss_fn = nn.BCEWithLogitsLoss()
            elif loss_type == "bce":
                loss_fn = nn.BCELoss()
            
            # Multi-class Classification
            elif loss_type == "cross_entropy" or loss_type == "crossentropyloss":
                loss_fn = nn.CrossEntropyLoss()
            
            # Regression
            elif loss_type == "mse" or loss_type == "mseloss":
                loss_fn = nn.MSELoss()
            elif loss_type == "mae" or loss_type == "l1" or loss_type == "l1loss":
                loss_fn = nn.L1Loss()
            elif loss_type == "smooth_l1" or loss_type == "smoothl1loss":
                loss_fn = nn.SmoothL1Loss()
            
            # Default
            else:
                loss_fn = nn.CrossEntropyLoss()
            
            self.set_output_val(0, loss_fn)
        except Exception as e:
                self.set_output_val(0, None)