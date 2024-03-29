import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if 'encoded_spconv_tensor' in batch_dict.keys():
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            encoded_spatial_stride = batch_dict['encoded_spconv_tensor_stride']
            spatial_features = encoded_spconv_tensor.dense()
            if len(spatial_features.size()) == 5:  # CHK MARK, no need for 2D shape
                N, C, D, H, W = spatial_features.shape  # (N, C, Z, Y, X)
                spatial_features = spatial_features.view(N, C * D, H, W)
        elif 'encoded_tensor' in batch_dict.keys():  # CHK MARK, no need for dense tensor
            encoded_spatial_stride = batch_dict['encoded_tensor_stride']
            spatial_features = batch_dict['encoded_tensor']
        else:
            raise NotImplementedError

        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = encoded_spatial_stride
        return batch_dict
