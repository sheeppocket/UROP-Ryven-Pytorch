


from ryven.node_env import *
import numpy as np


class NumpyToTensor(Node):
    """
    Converts numpy array to torch tensor using torch.from_numpy.
    """
    title = "Numpy to Tensor"
    init_inputs = [
        NodeInputType(label="numpy_array")
    ]
    init_outputs = [
        NodeOutputType(label="tensor")
    ]

    def update_event(self):
        import torch
        data = self.input(0)

        if data is None:
            return

        try:
            # Handle pandas DataFrame
            if hasattr(data, 'values'):
                data = data.values

            # Convert to numpy if not already
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            # Ensure array is contiguous for torch.from_numpy
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            # Convert to torch tensor
            import torch
            tensor = torch.from_numpy(data)
            self.set_output_val(0, tensor)
        except Exception as e:
            self.set_output_val(0, None)
            print(f"Error converting to tensor: {e}")


class DataframeToTensorDataset(Node):
    """
    Converts pandas DataFrame to torch TensorDataset.
    Separates features and labels.
    """
    title = "DataFrame to TensorDataset"
    init_inputs = [
        NodeInputType(label="dataframe"),
        NodeInputType(label="label_column")
    ]
    init_outputs = [
        NodeOutputType(label="dataset")
    ]

    def update_event(self):
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        df = self.input(0)
        label_col = self.input(1)

        if df is None:
            return

        try:
            import torch
            from torch.utils.data import TensorDataset as TorchTensorDataset
            # Extract features and labels
            if label_col is not None:
                if label_col not in df.columns:
                    print(f"Error: Column '{label_col}' not found in DataFrame")
                    self.set_output_val(0, None)
                    return
                labels = df[label_col].values
                features = df.drop(columns=[label_col]).values
            else:
                features = df.values
                labels = None

            # Ensure arrays are contiguous
            features = np.ascontiguousarray(features)
            
            # Convert to torch tensors
            features_tensor = torch.from_numpy(features).float()
            
            if labels is not None:
                labels = np.ascontiguousarray(labels)
                labels_tensor = torch.from_numpy(labels.astype(np.int64))
                dataset = TorchTensorDataset(features_tensor, labels_tensor)
            else:
                dataset = TorchTensorDataset(features_tensor)

            self.set_output_val(0, dataset)
        except Exception as e:
            self.set_output_val(0, None)
            print(f"Error converting DataFrame to TensorDataset: {e}")


class NumpyArrayToTensorDataset(Node):
    """
    Converts numpy arrays to torch TensorDataset.
    Supports multiple input arrays.
    """
    title = "Numpy to TensorDataset"
    init_inputs = [
        NodeInputType(label="arrays")
    ]
    init_outputs = [
        NodeOutputType(label="dataset")
    ]

    def update_event(self):
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        arrays = self.input(0)

        if arrays is None:
            return

        try:
            import torch
            from torch.utils.data import TensorDataset as TorchTensorDataset
            # Ryven may pass single array instead of list
            if not isinstance(arrays, (list, tuple)):
                arrays = [arrays]

            # Convert all arrays to torch tensors
            tensors = []
            for arr in arrays:
                if hasattr(arr, 'values'):  # pandas Series/DataFrame
                    arr = arr.values
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr)
                
                # Ensure array is contiguous
                if not arr.flags['C_CONTIGUOUS']:
                    arr = np.ascontiguousarray(arr)
                
                tensors.append(torch.from_numpy(arr))

            dataset = TorchTensorDataset(*tensors)
            self.set_output_val(0, dataset)
        except Exception as e:
            self.set_output_val(0, None)
            print(f"Error converting arrays to TensorDataset: {e}")
