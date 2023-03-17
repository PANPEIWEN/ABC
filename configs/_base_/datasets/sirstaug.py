# dataset settings
data = dict(
    dataset_type='SIRSTAUG',
    data_root='/data1/ppw/works/All_ISTD/datasets/SIRST_AUG',
    base_size=256,
    crop_size=256,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=32,
    test_batch=32,
    train_dir='trainval',
    test_dir='test'
)
