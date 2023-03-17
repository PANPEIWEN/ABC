# optimizer
optimizer = dict(
    type='SGD',
    setting=dict(lr=0.01, weight_decay=0.0005)
)
optimizer_config = dict()
# learning policy
# TODO warmup only 'linear'
lr_config = dict(policy='PolyLR', warmup='linear', power=0.9, min_lr=1e-4, warmup_epochs=5)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=False, interval=1)
evaluation = dict(epochval=1, metric='mIoU', pre_eval=True)
