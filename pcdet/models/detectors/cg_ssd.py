from .detector3d_template import Detector3DTemplate


class CGSSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_cgam, tb_dict_cgam = self.dense_head.aux_module.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict,
            **tb_dict_cgam,
        }

        loss = loss_rpn + loss_cgam
        return loss, tb_dict, disp_dict