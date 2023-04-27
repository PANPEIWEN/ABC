_base_ = [
    'abcnet_clft-s_256x256_1500e_nudt.py'
]

model = dict(
    decode_head=dict(
        dim=64
    )
)
data = dict(train_batch=16)
