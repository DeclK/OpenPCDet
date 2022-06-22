## Dataset Organization
```
- OpenPCDet
    - data
        - aw
            - data
            - ImageSets
            -------------------------------
            - aw_infos_train/val.pkl
            - ...
```

## Create Dataset
```shell
python -m pcdet.datasets.aw.aw_dataset --func create_aw_infos \
    --cfg_file tools/cfgs/dataset_configs/aw_dataset.yaml \
```