model = dict(
    name='ABCNet',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None
    ),
    decode_head=dict(
        type='ABCNet',
        in_ch=3,
        out_ch=1,
        dim=64, # in dim
        ori_h=256, # image height == width
        deep_supervision=True
    ),
    loss=dict(type='SoftIoULoss')
)
