class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 5751
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    # /home/mathos/Documents/cs/bdrp/repos/arcface-pytorch
    train_root = './data/imgs_/'
    train_list = './lfw_test_pair.txt'
    val_list = './lfw_test_pair.txt'

    test_root = './data/imgs_/'
    test_list = './lfw_test_pair.txt'

    lfw_root = './data/imgs_/'
    lfw_test_list = './lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    # load_model_path = 'models/resnet18.pth'
    # test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 112, 112)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
