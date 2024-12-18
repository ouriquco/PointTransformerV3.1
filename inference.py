import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.test import TESTERS

def load_model():
    args = {
    'save_path': '/fs/atipa/data/cmpe258-sp24/fa24_team14/codys_workspace/PointTransformerV3.1/exp/waymo/semsegv3-waymo-depth-exp',
    'weight': '/fs/atipa/data/cmpe258-sp24/fa24_team14/codys_workspace/PointTransformerV3.1/exp/waymo/semsegv3-waymo-depth-exp/model/model_best.pth'
    }
    cfg = default_config_parser('/fs/atipa/data/cmpe258-sp24/fa24_team14/codys_workspace/PointTransformerV3.1/configs/waymo/semseg-pt-v3m1-0-base_depth_exp.py',args)
    cfg = default_setup(cfg)
    print(cfg)
    model = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))

    # Uses overidden test method from SemSegTester class
    model.test()
    return model.model
    # model = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    # model = model.model
    # checkpoint_data = torch.load('/fs/atipa/data/cmpe258-sp24/fa24_team14/codys_workspace/PointTransformerV3.1/exp/waymo/semsegV3_waymo_20_epochs_2/model/model_best.pth')
    # model.load_state_dict(checkpoint_data['state_dict'])
    # return model


def test():

    model = load_model()

test()