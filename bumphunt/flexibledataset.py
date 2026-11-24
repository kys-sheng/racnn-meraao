import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
import torch 

class FlexibleMLDataset(Dataset):
    def __init__(
        self,
        histograms: Dict[str, np.ndarray],  # e.g., {'hist1': (N, H1), 'hist2': (N, H2)}
        other_inputs: Dict[str, np.ndarray],
        flow_outputs: Dict[str, np.ndarray],
        mlp_outputs: Dict[str, np.ndarray],
        input_norm: Dict[str, str],
        output_norm: Dict[str, str],
        normalize_on_init: bool = True,  # Flag to control immediate normalization
        input_scalers : Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        output_scalers: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    ):
        self.histograms    = {k: v.astype(np.double) for k, v in histograms.items()}
        self.other_inputs  = {k: v.astype(np.double) for k, v in other_inputs.items()}
        self.flow_outputs  = {k: v.astype(np.double) for k, v in flow_outputs.items()}
        self.mlp_outputs   = {k: v.astype(np.double) for k, v in mlp_outputs.items()}

        self.input_norm    = input_norm
        self.output_norm   = output_norm

        assert all(i in self.histograms.keys() or i in self.other_inputs.keys() for i in self.input_norm.keys())
        assert all(i in self.flow_outputs.keys() or i in self.mlp_outputs.keys() for i in self.output_norm.keys())

        self.input_scalers  = input_scalers or {}
        self.output_scalers = output_scalers or {}

        self.normalize_on_init = normalize_on_init

        if self.normalize_on_init:
            self._fit_scalers()
            self._normalize_all()

    def _get_scaler(self, method: str):
        if method in ['global_minmax', 'local_minmax', 'standard', 'none']:
            return {'mean': None, 'std': None, 'min': None, 'max': None, 'method': method}
        raise ValueError(f"Unknown normalization method: {method}")

    def _fit_scalers(self):
        for key, data in self.histograms.items():
            method = self.input_norm.get(key)
            if method:
                scaler = self._get_scaler(method)
                if method == 'standard':
                    scaler['mean'] = np.mean(data, axis=0)
                    scaler['std'] = np.std(data, axis=0)
                elif method == 'local_minmax':
                    scaler['min'] = np.min(data, axis=0)
                    scaler['max'] = np.max(data, axis=0)
                elif method == 'global_minmax':
                    scaler['min'] = np.min(data)
                    scaler['max'] = np.max(data)
                elif method == 'none':
                    # No normalization - data stays as-is
                    pass
                self.input_scalers[key] = scaler
            else:
                raise KeyError(f"histograms ({key}) has no norm choices")

        for key, data in self.other_inputs.items():
            method = self.input_norm.get(key)
            if method:
                scaler = self._get_scaler(method)
                if method == 'standard':
                    scaler['mean'] = np.mean(data, axis=0)
                    scaler['std'] = np.std(data, axis=0)
                elif method == 'none':
                    # No normalization - data stays as-is
                    pass
                else:
                    scaler['min'] = np.min(data)
                    scaler['max'] = np.max(data)
                self.input_scalers[key] = scaler
            else:
                raise KeyError(f"other_inputs ({key}) has no norm choices")
                
        for key, data in self.flow_outputs.items():
            method = self.output_norm.get(key)
            if method:
                scaler = self._get_scaler(method)
                if method == 'standard':
                    scaler['mean'] = np.nanmean(data, axis=0)
                    scaler['std'] = np.nanstd(data, axis=0)
                elif method == 'none':
                    # No normalization - data stays as-is
                    pass
                else:
                    scaler['min'] = np.nanmin(data)
                    scaler['max'] = np.nanmax(data)
                self.output_scalers[key] = scaler
            else:
                raise KeyError(f"flow_outputs ({key}) has no norm choices")

        for key, data in self.mlp_outputs.items():
            method = self.output_norm.get(key)
            if method:
                scaler = self._get_scaler(method)
                if method == 'standard':
                    scaler['mean'] = np.mean(data, axis=0)
                    scaler['std'] = np.std(data, axis=0)
                elif method == 'none':
                    # No normalization - data stays as-is
                    pass
                else:
                    scaler['min'] = np.min(data)
                    scaler['max'] = np.max(data)
                self.output_scalers[key] = scaler
            else:
                raise KeyError(f"mlp_outputs ({key}) has no norm choices")

    def _normalize(self, data, scaler):
        if scaler['method'] == 'standard':
            assert scaler['mean'] is not None 
            assert scaler['std'] is not None 
            return (data - scaler['mean']) / (scaler['std'] + 1e-8)
        elif scaler['method'] == 'none':
            # No normalization - return data unchanged
            return data
        else:
            # global_minmax or local_minmax
            assert scaler['min'] is not None 
            assert scaler['max'] is not None 
            return (data - scaler['min']) / (scaler['max'] - scaler['min'] + 1e-8)

    def _unnormalize(self, data, scaler):
        if scaler['method'] == 'standard':
            assert scaler['mean'] is not None 
            assert scaler['std'] is not None 
            return data * (scaler['std'] + 1e-8) + scaler['mean']
        elif scaler['method'] == 'none':
            # No normalization was applied - return data unchanged
            return data
        else:
            assert scaler['min'] is not None 
            assert scaler['max'] is not None 
            return data * (scaler['max'] - scaler['min'] + 1e-8) + scaler['min']

    def _normalize_all(self):
        for key, data in self.histograms.items():
            if key in self.input_scalers:
                self.histograms[key] = self._normalize(data, self.input_scalers[key])

        for key, data in self.other_inputs.items():
            if key in self.input_scalers:
                self.other_inputs[key] = self._normalize(data, self.input_scalers[key])

        for key, data in self.flow_outputs.items():
            if key in self.output_scalers:
                self.flow_outputs[key] = self._normalize(data, self.output_scalers[key])

        for key, data in self.mlp_outputs.items():
            if key in self.output_scalers:
                self.mlp_outputs[key] = self._normalize(data, self.output_scalers[key])

    def normalize_batch(self, group: str, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        scalers = self.input_scalers if group in ['histograms', 'other_inputs'] else self.output_scalers
        return {
            key: self._normalize(val, scalers[key]) if key in scalers else val
            for key, val in batch.items()
        }

    def unnormalize_batch(self, group: str, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        scalers = self.input_scalers if group in ['histograms', 'other_inputs'] else self.output_scalers
        return {
            key: self._unnormalize(val, scalers[key]) if key in scalers else val
            for key, val in batch.items()
        }

    def apply_to_test_data(self, test_data: 'FlexibleMLDataset'):
        # Apply stored statistics (mean, std, min, max) to normalize test data
        for group, scalers in [('histograms', self.input_scalers), ('other_inputs', self.input_scalers), 
                               ('flow_outputs', self.output_scalers), ('mlp_outputs', self.output_scalers)]:
            for key, scaler in scalers.items():
                if key in test_data.__dict__[group]:
                    test_data.__dict__[group][key] = self._normalize(test_data.__dict__[group][key], scaler)
    
    def unnormalize_input(self, key: str, data: np.ndarray) -> np.ndarray:
        scaler = self.input_scalers.get(key)
        return self._unnormalize(data, scaler) #if scaler else data

    def unnormalize_output(self, key_or_group: str, data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray] or np.ndarray:
        if data is not None:
            scaler = self.output_scalers.get(key_or_group)
            return self._unnormalize(data, scaler) #if scaler else data
        else:
            # unnormalize all outputs for a group
            result = {}
            group_data = self.flow_outputs if key_or_group == 'flow_outputs' else self.mlp_outputs
            for key, val in group_data.items():
                scaler = self.output_scalers.get(key)
                result[key] = self._unnormalize(val, scaler) #if scaler else val
            return result

    def __len__(self):
        first_key = next(iter(self.histograms))
        return len(self.histograms[first_key])

    def __getitem__(self, idx):
        x_hist = torch.hstack([torch.tensor(self.histograms[k][idx], dtype=torch.float32) for k in self.histograms]).unsqueeze(0)  # (1, total_hist_bins)
        x_aux  = torch.tensor([self.other_inputs[k][idx] for k in self.other_inputs], dtype=torch.float32)  # (num_aux_features,)
        y_flow = torch.tensor([self.flow_outputs[k][idx] for k in self.flow_outputs], dtype=torch.float32)  # (num_flow_outputs,)
        y_mlp  = torch.tensor([self.mlp_outputs[k][idx] for k in self.mlp_outputs], dtype=torch.float32)    # (num_mlp_outputs,)

        return x_hist, x_aux, y_flow, y_mlp

    def split_train_val(self, val_ratio=0.2, seed=None) -> Tuple['FlexibleMLDataset', 'FlexibleMLDataset']:
        idx = np.arange(len(self))
        train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed)

        def subset_dict(d: Dict[str, np.ndarray], indices: np.ndarray):
            return {k: v[indices] for k, v in d.items()}

        train_data = FlexibleMLDataset(
            histograms=subset_dict(self.histograms, train_idx),
            other_inputs=subset_dict(self.other_inputs, train_idx),
            flow_outputs=subset_dict(self.flow_outputs, train_idx),
            mlp_outputs=subset_dict(self.mlp_outputs, train_idx),
            input_norm=self.input_norm,
            output_norm=self.output_norm,
            normalize_on_init = False,
            input_scalers = self.input_scalers,
            output_scalers = self.output_scalers
        )

        val_data = FlexibleMLDataset(
            histograms=subset_dict(self.histograms, val_idx),
            other_inputs=subset_dict(self.other_inputs, val_idx),
            flow_outputs=subset_dict(self.flow_outputs, val_idx),
            mlp_outputs=subset_dict(self.mlp_outputs, val_idx),
            input_norm=self.input_norm,
            output_norm=self.output_norm,
            normalize_on_init = False,
            input_scalers = self.input_scalers,
            output_scalers = self.output_scalers
        )

        return train_data, val_data
    
    def subset_by_multiple_conditions(
        self, 
        group: str, 
        conditions: Dict[str, callable], 
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        
        if group == "histograms":
            group_data = self.histograms
            scalers = self.input_scalers
        elif group == "other_inputs":
            group_data = self.other_inputs
            scalers = self.input_scalers
        elif group == "flow_outputs":
            group_data = self.flow_outputs
            scalers = self.output_scalers
        elif group == "mlp_outputs":
            group_data = self.mlp_outputs
            scalers = self.output_scalers
        else:
            raise ValueError(f"Unknown group: {group}")

        # Unnormalize group data
        unnormalized_group = {
            key: self._unnormalize(data, scalers[key]) if key in scalers else data
            for key, data in group_data.items()
        }

        mask = np.ones(len(self), dtype=bool)
        # Apply group conditions
        for key, cond_fn in conditions.items():
            data = unnormalized_group[key]
            cond_result = cond_fn(data)
            if cond_result.shape != mask.shape:
                raise ValueError(f"Condition for key '{key}' did not return a mask of the correct shape")
            mask &= cond_result

        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return {}, {}, {}, {}

        def subset_dict(d): return {k: v[indices] for k, v in d.items() if len(v) > 0}

        x_hist     = subset_dict(self.histograms)
        x_others   = subset_dict(self.other_inputs)
        y_flow     = subset_dict(self.flow_outputs)
        y_mlp      = subset_dict(self.mlp_outputs)

        return x_hist, x_others, y_flow, y_mlp


        
# class FlexibleMLDataset(Dataset):
#     def __init__(
#         self,
#         histograms: Dict[str, np.ndarray],  # e.g., {'hist1': (N, H1), 'hist2': (N, H2)}
#         other_inputs: Dict[str, np.ndarray],
#         flow_outputs: Dict[str, np.ndarray],
#         mlp_outputs: Dict[str, np.ndarray],
#         input_norm: Dict[str, str],
#         output_norm: Dict[str, str],
#         normalize_on_init: bool = True,  # Flag to control immediate normalization
#         input_scalers : Optional[Dict[str, Dict[str, np.ndarray]]] = None,
#         output_scalers: Optional[Dict[str, Dict[str, np.ndarray]]] = None

#     ):
#         self.histograms    = {k: v.astype(np.double) for k, v in histograms.items()}
#         self.other_inputs  = {k: v.astype(np.double) for k, v in other_inputs.items()}
#         self.flow_outputs  = {k: v.astype(np.double) for k, v in flow_outputs.items()}
#         self.mlp_outputs   = {k: v.astype(np.double) for k, v in mlp_outputs.items()}

#         self.input_norm    = input_norm
#         self.output_norm   = output_norm

#         assert all(i in self.histograms.keys() or i in self.other_inputs.keys() for i in self.input_norm.keys())
#         assert all(i in self.flow_outputs.keys() or i in self.mlp_outputs.keys() for i in self.output_norm.keys())

#         self.input_scalers  = input_scalers or {}
#         self.output_scalers = output_scalers or {}

#         self.normalize_on_init = normalize_on_init

#         if self.normalize_on_init:
#             self._fit_scalers()
#             self._normalize_all()

#     def _get_scaler(self, method: str):
#         if method in ['global_minmax', 'local_minmax', 'standard']:
#             return {'mean': None, 'std': None, 'min': None, 'max': None, 'method': method}
#         raise ValueError(f"Unknown normalization method: {method}")

#     def _fit_scalers(self):
#         for key, data in self.histograms.items():
#             method = self.input_norm.get(key)
#             if method:
#                 scaler = self._get_scaler(method)
#                 if method == 'standard':
#                     scaler['mean'] = np.mean(data, axis=0)
#                     scaler['std'] = np.std(data, axis=0)
#                 elif method == 'local_minmax':
#                     scaler['min'] = np.min(data, axis=0)
#                     scaler['max'] = np.max(data, axis=0)
#                 elif method == 'global_minmax':
#                     scaler['min'] = np.min(data)
#                     scaler['max'] = np.max(data)
#                 self.input_scalers[key] = scaler
#             else:
#                 raise KeyError(f"histograms ({key}) has no norm choices")

#         for key, data in self.other_inputs.items():
#             method = self.input_norm.get(key)
#             if method:
#                 scaler = self._get_scaler(method)
#                 if method == 'standard':
#                     scaler['mean'] = np.mean(data, axis=0)
#                     scaler['std'] = np.std(data, axis=0)
#                 else:
#                     scaler['min'] = np.min(data)
#                     scaler['max'] = np.max(data)
#                 self.input_scalers[key] = scaler
#             else:
#                 raise KeyError(f"other_inputs ({key}) has no norm choices")
                
#         for key, data in self.flow_outputs.items():
#             method = self.output_norm.get(key)
#             if method:
#                 scaler = self._get_scaler(method)
#                 if method == 'standard':
#                     scaler['mean'] = np.nanmean(data, axis=0)
#                     scaler['std'] = np.nanstd(data, axis=0)
#                 else:
#                     scaler['min'] = np.nanmin(data)
#                     scaler['max'] = np.nanmax(data)
#                 self.output_scalers[key] = scaler
#             else:
#                 raise KeyError(f"flow_outputs ({key}) has no norm choices")


#         for key, data in self.mlp_outputs.items():
#             method = self.output_norm.get(key)
#             if method:
#                 scaler = self._get_scaler(method)
#                 if method == 'standard':
#                     scaler['mean'] = np.mean(data, axis=0)
#                     scaler['std'] = np.std(data, axis=0)
#                 else:
#                     scaler['min'] = np.min(data)
#                     scaler['max'] = np.max(data)
#                 self.output_scalers[key] = scaler
#             else:
#                 raise KeyError(f"mlp_outputs ({key}) has no norm choices")

#     def _normalize(self, data, scaler):
#         if scaler['method'] == 'standard':
#             assert scaler['mean'] is not None 
#             assert scaler['std'] is not None 
#             return (data - scaler['mean']) / (scaler['std'] + 1e-8)
#         else:
#             assert scaler['min'] is not None 
#             assert scaler['max'] is not None 
#             return (data - scaler['min']) / (scaler['max'] - scaler['min'] + 1e-8)

#     def _unnormalize(self, data, scaler):
#         if scaler['method'] == 'standard':
#             assert scaler['mean'] is not None 
#             assert scaler['std'] is not None 
#             return data * (scaler['std'] + 1e-8) + scaler['mean']
#         else:
#             assert scaler['min'] is not None 
#             assert scaler['max'] is not None 
#             return data * (scaler['max'] - scaler['min'] + 1e-8) + scaler['min']

#     def _normalize_all(self):
#         for key, data in self.histograms.items():
#             if key in self.input_scalers:
#                 self.histograms[key] = self._normalize(data, self.input_scalers[key])

#         for key, data in self.other_inputs.items():
#             if key in self.input_scalers:
#                 self.other_inputs[key] = self._normalize(data, self.input_scalers[key])

#         for key, data in self.flow_outputs.items():
#             if key in self.output_scalers:
#                 self.flow_outputs[key] = self._normalize(data, self.output_scalers[key])

#         for key, data in self.mlp_outputs.items():
#             if key in self.output_scalers:
#                 self.mlp_outputs[key] = self._normalize(data, self.output_scalers[key])

#     def normalize_batch(self, group: str, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
#         scalers = self.input_scalers if group in ['histograms', 'other_inputs'] else self.output_scalers
#         return {
#             key: self._normalize(val, scalers[key]) if key in scalers else val
#             for key, val in batch.items()
#         }

#     def unnormalize_batch(self, group: str, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
#         scalers = self.input_scalers if group in ['histograms', 'other_inputs'] else self.output_scalers
#         return {
#             key: self._unnormalize(val, scalers[key]) if key in scalers else val
#             for key, val in batch.items()
#         }

#     def apply_to_test_data(self, test_data: 'FlexibleMLDataset'):
#         # Apply stored statistics (mean, std, min, max) to normalize test data
#         for group, scalers in [('histograms', self.input_scalers), ('other_inputs', self.input_scalers), 
#                                ('flow_outputs', self.output_scalers), ('mlp_outputs', self.output_scalers)]:
#             for key, scaler in scalers.items():
#                 if key in test_data.__dict__[group]:
#                     test_data.__dict__[group][key] = self._normalize(test_data.__dict__[group][key], scaler)
    
#     def unnormalize_input(self, key: str, data: np.ndarray) -> np.ndarray:
#         scaler = self.input_scalers.get(key)
#         return self._unnormalize(data, scaler) #if scaler else data

#     def unnormalize_output(self, key_or_group: str, data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray] or np.ndarray:
#         if data is not None:
#             scaler = self.output_scalers.get(key_or_group)
#             return self._unnormalize(data, scaler) #if scaler else data
#         else:
#             # unnormalize all outputs for a group
#             result = {}
#             group_data = self.flow_outputs if key_or_group == 'flow_outputs' else self.mlp_outputs
#             for key, val in group_data.items():
#                 scaler = self.output_scalers.get(key)
#                 result[key] = self._unnormalize(val, scaler) #if scaler else val
#             return result

#     def __len__(self):
#         first_key = next(iter(self.histograms))
#         return len(self.histograms[first_key])

#     # def __getitem__(self, idx):
#     #     x_hist    = {k: v[idx] for k, v in self.histograms.items()}
#     #     x_others  = {k: v[idx] for k, v in self.other_inputs.items()}
#     #     y_flow    = {k: v[idx] for k, v in self.flow_outputs.items()}
#     #     y_mlp     = {k: v[idx] for k, v in self.mlp_outputs.items()}
#     #     return x_hist, x_others, y_flow, y_mlp
#     def __getitem__(self, idx):
#         x_hist = torch.hstack([torch.tensor(self.histograms[k][idx], dtype=torch.float32) for k in self.histograms]).unsqueeze(0)  # (1, total_hist_bins)
#         x_aux  = torch.tensor([self.other_inputs[k][idx] for k in self.other_inputs], dtype=torch.float32)  # (num_aux_features,)
#         y_flow = torch.tensor([self.flow_outputs[k][idx] for k in self.flow_outputs], dtype=torch.float32)  # (num_flow_outputs,)
#         y_mlp  = torch.tensor([self.mlp_outputs[k][idx] for k in self.mlp_outputs], dtype=torch.float32)    # (num_mlp_outputs,)

#         return x_hist, x_aux, y_flow, y_mlp


#     def split_train_val(self, val_ratio=0.2, seed=None) -> Tuple['FlexibleMLDataset', 'FlexibleMLDataset']:
#         idx = np.arange(len(self))
#         train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed)

#         def subset_dict(d: Dict[str, np.ndarray], indices: np.ndarray):
#             return {k: v[indices] for k, v in d.items()}

#         train_data = FlexibleMLDataset(
#             histograms=subset_dict(self.histograms, train_idx),
#             other_inputs=subset_dict(self.other_inputs, train_idx),
#             flow_outputs=subset_dict(self.flow_outputs, train_idx),
#             mlp_outputs=subset_dict(self.mlp_outputs, train_idx),
#             input_norm=self.input_norm,
#             output_norm=self.output_norm,
#             normalize_on_init = False,
#             input_scalers = self.input_scalers,
#             output_scalers = self.output_scalers

#         )

#         val_data = FlexibleMLDataset(
#             histograms=subset_dict(self.histograms, val_idx),
#             other_inputs=subset_dict(self.other_inputs, val_idx),
#             flow_outputs=subset_dict(self.flow_outputs, val_idx),
#             mlp_outputs=subset_dict(self.mlp_outputs, val_idx),
#             input_norm=self.input_norm,
#             output_norm=self.output_norm,
#             normalize_on_init = False,
#             input_scalers = self.input_scalers,
#             output_scalers = self.output_scalers

#         )

#         return train_data, val_data
    
#     # def subset_by_multiple_conditions(self, group: str, conditions: Dict[str, callable]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
#     #     group_data = self.flow_outputs if group == "flow_outputs" else self.mlp_outputs
#     #     assert group_data is not None

#     #     # Unnormalize outputs
#     #     unnormalized_data = {
#     #         key: self._unnormalize(data, self.output_scalers[key]) if key in self.output_scalers else data
#     #         for key, data in group_data.items()
#     #     }

#     #     mask = np.ones(len(self), dtype=bool)
#     #     for key, cond_fn in conditions.items():
#     #         data = unnormalized_data[key]
#     #         cond_result = cond_fn(data)
#     #         if cond_result.shape != mask.shape:
#     #             raise ValueError(f"Condition for key '{key}' did not return a mask of the correct shape")
#     #         mask &= cond_result

#     #     indices = np.where(mask)[0]
        
#     #     if len(indices) == 0:
#     #         return {}, {}, {}, {}

#     #     def subset_dict(d): return {k: v[indices] for k, v in d.items() if len(v) > 0}

#     #     x_hist     = subset_dict(self.histograms)
#     #     x_others   = subset_dict(self.other_inputs)
#     #     y_flow     = subset_dict(self.flow_outputs)
#     #     y_mlp      = subset_dict(self.mlp_outputs)

#     #     return x_hist, x_others, y_flow, y_mlp
    
#     def subset_by_multiple_conditions(
#         self, 
#         group: str, 
#         conditions: Dict[str, callable], 
#     ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        
#         if group == "histograms":
#             group_data = self.histograms
#             scalers = self.input_scalers
#         elif group == "other_inputs":
#             group_data = self.other_inputs
#             scalers = self.input_scalers
#         elif group == "flow_outputs":
#             group_data = self.flow_outputs
#             scalers = self.output_scalers
#         elif group == "mlp_outputs":
#             group_data = self.mlp_outputs
#             scalers = self.output_scalers
#         else:
#             raise ValueError(f"Unknown group: {group}")

#         # Unnormalize group data
#         unnormalized_group = {
#             key: self._unnormalize(data, scalers[key]) if key in scalers else data
#             for key, data in group_data.items()
#         }

#         mask = np.ones(len(self), dtype=bool)
#         # Apply group conditions
#         for key, cond_fn in conditions.items():
#             data = unnormalized_group[key]
#             cond_result = cond_fn(data)
#             if cond_result.shape != mask.shape:
#                 raise ValueError(f"Condition for key '{key}' did not return a mask of the correct shape")
#             mask &= cond_result

#         indices = np.where(mask)[0]
        
#         if len(indices) == 0:
#             return {}, {}, {}, {}

#         def subset_dict(d): return {k: v[indices] for k, v in d.items() if len(v) > 0}

#         x_hist     = subset_dict(self.histograms)
#         x_others   = subset_dict(self.other_inputs)
#         y_flow     = subset_dict(self.flow_outputs)
#         y_mlp      = subset_dict(self.mlp_outputs)

#         return x_hist, x_others, y_flow, y_mlp    
     


# ---------------------------------------------------------------------------
# HDF5 storage (public release format)
#
# A FlexibleMLDataset is fully described by four dicts of named arrays plus
# its normalisation settings and fitted scalers. save_h5 writes exactly that
# state; load_h5 rebuilds the object without re-normalising, so a round trip
# is bit-identical. Arrays are stored as they live in the object, i.e. already
# normalised; use ds.unnormalize_input / ds.unnormalize_output for physical
# values.
#
# Layout:
#   /histograms/<key>, /other_inputs/<key>, /flow_outputs/<key>, /mlp_outputs/<key>
#       (each group has attr "keys": JSON list giving the column order)
#   /scalers/input/<key>, /scalers/output/<key>
#       (attr "method"; datasets "mean", "std", "min", "max" where not None)
#   root attrs: format, format_version, n_samples, input_norm, output_norm,
#               normalize_on_init
# ---------------------------------------------------------------------------
import json as _json

_H5_GROUPS = ("histograms", "other_inputs", "flow_outputs", "mlp_outputs")
_H5_FORMAT = "racnn-FlexibleMLDataset"


def save_h5(ds, path, compression="gzip", compression_opts=4):
    """Write a FlexibleMLDataset to an HDF5 file."""
    import h5py
    with h5py.File(path, "w") as f:
        f.attrs["format"] = _H5_FORMAT
        f.attrs["format_version"] = 1
        f.attrs["n_samples"] = len(ds)
        f.attrs["input_norm"] = _json.dumps(ds.input_norm)
        f.attrs["output_norm"] = _json.dumps(ds.output_norm)
        f.attrs["normalize_on_init"] = bool(ds.normalize_on_init)
        for g in _H5_GROUPS:
            d = getattr(ds, g)
            grp = f.create_group(g)
            grp.attrs["keys"] = _json.dumps(list(d.keys()))
            for k, v in d.items():
                v = np.asarray(v)
                kw = dict(compression=compression, compression_opts=compression_opts, shuffle=True) if v.ndim and v.size > 1024 else {}
                grp.create_dataset(k, data=v, **kw)
        for side, scalers in (("input", ds.input_scalers), ("output", ds.output_scalers)):
            sg = f.create_group(f"scalers/{side}")
            sg.attrs["keys"] = _json.dumps(list(scalers.keys()))
            for k, s in scalers.items():
                kg = sg.create_group(k)
                kg.attrs["method"] = s["method"]
                for field in ("mean", "std", "min", "max"):
                    if s.get(field) is not None:
                        kg.create_dataset(field, data=np.asarray(s[field]))


def load_h5(path, index=None):
    """Read a FlexibleMLDataset written by save_h5.

    index: optional increasing array of row indices to load only a subset
    (useful for the large mono-jet files); scalers are kept unchanged.
    """
    import h5py
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != _H5_FORMAT:
            raise ValueError(f"{path}: not a {_H5_FORMAT} file")
        groups = {}
        for g in _H5_GROUPS:
            grp = f[g]
            groups[g] = {k: (grp[k][()] if index is None else grp[k][np.asarray(index)]) for k in _json.loads(grp.attrs["keys"])}
        scalers = {}
        for side in ("input", "output"):
            sg = f[f"scalers/{side}"]
            scalers[side] = {}
            for k in _json.loads(sg.attrs["keys"]):
                kg = sg[k]
                s = {"mean": None, "std": None, "min": None, "max": None, "method": kg.attrs["method"]}
                for field in ("mean", "std", "min", "max"):
                    if field in kg:
                        s[field] = kg[field][()]
                scalers[side][k] = s
        ds = FlexibleMLDataset(
            histograms=groups["histograms"],
            other_inputs=groups["other_inputs"],
            flow_outputs=groups["flow_outputs"],
            mlp_outputs=groups["mlp_outputs"],
            input_norm=_json.loads(f.attrs["input_norm"]),
            output_norm=_json.loads(f.attrs["output_norm"]),
            normalize_on_init=False,
            input_scalers=scalers["input"],
            output_scalers=scalers["output"],
        )
        ds.normalize_on_init = bool(f.attrs["normalize_on_init"])
    return ds


def load_dataset(path):
    """Load a dataset from .h5 (release format) or a legacy pickled .pt.

    `path` may also be given without extension, in which case `<path>.h5`
    is preferred and `<path>.pt` is the fallback.
    """
    import os
    if not path.endswith((".h5", ".pt")):
        path = path + ".h5" if os.path.exists(path + ".h5") else path + ".pt"
    if path.endswith(".h5"):
        return load_h5(path)
    return torch.load(path, weights_only=False)
