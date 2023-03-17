_base_ = [
    'abcnet_clft-s_512x512_1500e_nuaa.py'
]

model = dict(
    decode_head=dict(
        dim=64
    )
)
data = dict(train_batch=4)