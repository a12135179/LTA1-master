"""
MODIFICATION NOTICES
THIS NOTICE IS BASED ON LICENSE_RESA
THIS FILE HAS BEEN MODIFIED 
ORIGIN CODE: https://github.com/ZJULearning/resa.git
"""
net = dict(
    type='RESANet',
)

backbone = dict( #backbone已經被修改 所以這個不能拿來參考
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    fea_stride=8,
)

resa = dict(
    type='RESA',
    alpha=2.0,
    iter=5,
    input_channel=256,
    conv_stride=9,
)

decoder = 'BUSD'        

trainer = dict(
    type='RESA'
)

evaluator = dict(
    type='Tusimple',        
    thresh = 0.60
)

optimizer = dict(
  type='sgd',
  lr=0.020,
  weight_decay=1e-4,
  momentum=0.9
)

total_iter = 80000
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 384
img_width = 640
cut_height = 160
seg_label = "seg_label"

dataset_path = './data/'
test_json_file = './data/test_label.json'

dataset = dict(
    train=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='train_val_gt.txt',
    ),
    val=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='test_gt.txt'
    ),
    test=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='test_gt.txt'
    )
)


loss_type = 'cross_entropy'
seg_loss_weight = 1.0


batch_size = 1
workers = 0
num_classes = 6 + 1
ignore_label = 255
epochs = 2
log_interval = 100
eval_ep = 1
save_ep = epochs
log_note = ''