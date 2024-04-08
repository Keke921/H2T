from pathlib import Path
from yacs.config import CfgNode as CN
import os
import time
import logging

_C = CN()
_C.name = ''
_C.print_freq = 40
_C.workers = 16
_C.log_dir = 'logs'
_C.model_dir = 'ckps'

# ----- DATASET BUILDER -----
_C.dataset = 'cifar10'
_C.data_path = './data/cifar10'
_C.num_classes = 100
_C.imb_factor = 0.01

_C.sampler = CN()
_C.sampler.type = 'default'
_C.sampler.weighted_sampler = CN()
_C.sampler.weighted_sampler.type = "balance"

_C.sampler.dual_sample = CN()
_C.sampler.dual_sample.enable = False
_C.sampler.dual_sample.type = 'long-tailed'

# ----- BACKBONE BUILDER -----
_C.backbone = 'resnet32'
_C.classifier = 'Classifier'
_C.ResLT = CN()
_C.ResLT.gamma = 0.7
_C.resume = ''
_C.head_class_idx = [0, 1]
_C.med_class_idx = [0, 1]
_C.tail_class_idx = [0, 1]

_C.use_norm = False
_C.num_experts = 1

_C.deterministic = True
_C.gpu = 0
_C.world_size = -1
_C.rank = -1
_C.dist_url = 'tcp://224.66.41.62:23456'
_C.dist_backend = 'nccl'
_C.multiprocessing_distributed = False
_C.distributed = False

# ----- LOSS BUILDER -----
_C.loss = CN()
_C.loss.type = "CrossEntropyLoss"
_C.loss.add_extra_info = False
_C.loss.CE = CN()
_C.loss.CE.reweight_CE = False

_C.loss.RIDE = CN()
_C.loss.RIDE.base_diversity_temperature = 1.0
_C.loss.RIDE.max_m = 0.5
_C.loss.RIDE.s = 30
_C.loss.RIDE.reweight = False
_C.loss.RIDE.reweight_epoch = -1
_C.loss.RIDE.base_loss_factor = 1.0
_C.loss.RIDE.additional_diversity_factor = -0.2
_C.loss.RIDE.reweight_factor = 0.05

_C.loss.LDAM = CN()
_C.loss.LDAM.max_m = 0.5
_C.loss.LDAM.s = 30
_C.loss.LDAM.reweight_epoch=-1

_C.loss.ResLT = CN()
_C.loss.ResLT.smooth = 0.0
_C.loss.ResLT.beta = 0.9900

_C.lr_scheduler = CN()
_C.lr_scheduler.type = "multistep"
_C.lr_scheduler.lr_step = [160, 180]
_C.lr_scheduler.lr_factor  = 0.1
_C.lr = 0.1
_C.batch_size = 128
_C.weight_decay = 0.002
_C.num_epochs = 200
_C.momentum = 0.9

_C.mode = None
_C.smooth_tail = None
_C.smooth_head = None
_C.shift_bn = False
_C.mixup = True
_C.alpha = 1.0
_C.H2T = -1.0


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()

def create_logger(cfg, cfg_name):    
    time_str = time.strftime('%Y%m%d%H%M')

    cfg_name = os.path.basename(cfg.name) #.split('.')[0]

    log_dir = Path("saved")  / (cfg_name + '_' + time_str) / Path('logs')
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    
    log_file = '{}.txt'.format(cfg_name)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir =  Path("saved") / (cfg_name + '_' + time_str) / Path('ckps')
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    code_dir = Path("saved") / (cfg_name + '_' + time_str) / Path('codes')
    print("=> code will be saved in {}".format(code_dir)) 
    code_dir.mkdir(parents=True, exist_ok=True)    

    return logger, str(model_dir), str(code_dir)