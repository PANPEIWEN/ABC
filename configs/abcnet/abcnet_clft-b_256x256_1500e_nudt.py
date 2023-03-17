_base_ = [
    'abcnet_clft-s_256x256_500e_nudt.py'
]

model = dict(
    decode_head=dict(
        dim=32
    )
)
data = dict(train_batch=16)