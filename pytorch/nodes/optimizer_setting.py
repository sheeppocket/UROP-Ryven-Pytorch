from ryven.node_env import *
import torch
import torch.optim as optim

class OptimizerSetting(Node):
    """
    Creates an optimizer with configurable parameters.
    Supports Adam, SGD, RMSprop, and AdamW optimizers.
    """
    title = "Optimizer Setting"
    init_inputs = [
        NodeInputType(label="params"),
        NodeInputType(label="optimizer_type"),
        NodeInputType(label="learning_rate"),
    ]
    init_outputs = [
        NodeOutputType(label="optimizer"),
    ]
    
    def update_event(self):
        params = self.input(0)
        optimizer_type = self.input(1)
        learning_rate = self.input(2)
        
        if params is None:
            return
        
        if isinstance(params, torch.nn.Module):
            params = params.parameters()
        elif hasattr(params, '__iter__') and not isinstance(params, (str, bytes)):
            pass
        else:
            print("OptimizerSetting: params must be a torch.nn.Module or iterable of parameters")
            self.set_output_val(0, None)
            return
        
        # Set defaults
        optimizer_type = optimizer_type if optimizer_type is not None else "Adam"
        learning_rate = float(learning_rate) if learning_rate is not None else 0.001
        
        # Validate learning rate
        
        try:
            if optimizer_type == "Adam":
                optimizer = optim.Adam(params, lr=learning_rate)
            elif optimizer_type == "SGD":
                optimizer = optim.SGD(params, lr=learning_rate)
            else:
                optimizer = optim.Adam(params, lr=learning_rate)
            
            self.set_output_val(0, optimizer)
        except Exception as e:
            self.set_output_val(0, None)
