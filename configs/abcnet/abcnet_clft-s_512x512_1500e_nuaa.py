_base_ = [
    '../_base_/datasets/nuaa.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/abcnet.py'
]

model = dict(
    decode_head=dict(
        dim=16,
        ori_h=512
    )
)

optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.0003, weight_decay=0.01, betas=(0.9, 0.999))
)
runner = dict(type='EpochBasedRunner', max_epochs=1500)
data = dict(train_batch=32)
