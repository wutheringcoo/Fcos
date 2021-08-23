from pathlib import Path
class Config:
    '''
        1) 定义类class A, class A(), classA(object) 三者等效
        2) Config = {'a' : 1 , 'b' : 2, ...}
        3) Config = dict(a=1, b=2, ...) 
    '''
    ############# backbone
    pretrained = True
    freeze_stage_1=True
    freeze_bn=True
    resnet50_weights = '/data-nbd/wuxc/Resource/weights/Restnets/resnet50-19c8e357.pth'
    # freeze_stage_1=True
    # freeze_bn=True

    ############# fpn
    features = 256
    # use_p5=True
    
    ############# head
    num_classes = 21
    use_GN_head = True
    prior = 0.01
    # add_centerness = True
    cnt_on_reg = False

    ############# train
    epochs = 32
    sample_ratio = 1.5
    train_path = 'VOC0712'
    strides = [8,16,32,64,128]
    window = [[0,64],[64,128],[128,256],[256,512],[512,999999]] 
    pos_sample_cls = True
    # train_split = "trainval"
    # train_split = "tmp_train"
    train_split = "test"
    save_dir = 'imgs'

    ############# test and inference
    class_nums = 21
    test_batch = 1
    test_path = 'VOC0712'
    test_split = "test"
    # test_split = "tmp_test"
    joint_centerness_prob = True
    max_pixels = 1000
    cls_score_threshold = 0.05
    iou_threshold = 0.6
    testresults_root = 'voc'

    ############# save
    save_dir = Path('/data-nbd/wuxc/FCOS_of/DDP/')
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'

