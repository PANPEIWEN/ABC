# dataset settings
data = dict(
    dataset_type='IRSTD1k',
    data_root='/data1/ppw/works/All_ISTD/datasets/IRSTD-1k',
    base_size=512,
    crop_size=512,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=16,
    test_batch=8,
    train_dir='trainval',
    test_dir='test'
)
