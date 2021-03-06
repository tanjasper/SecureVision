import argparse

def generate_parser():
    parser = argparse.ArgumentParser(description='FlatCam Face Verification')
    parser.add_argument('--dataA_dir', default='/media/hdd2/Datasets/ILSVRC2012/images/original',
                        help='directory to set A images (where we will train for anonymity')
    parser.add_argument('--dataB_dir', default='/media/hdd2/Datasets/vggface2',
                        help='directory to set B images')
    parser.add_argument('--filenamesA',
                        default='/media/hdd2/Datasets/ILSVRC2012/filenames/training_filenames_random_500k.txt',
                        help='training filenames for set A images')
    parser.add_argument('--filenamesB',
                        default='/media/hdd2/Datasets/vggface2/filenames/training_filenames_random_500k.txt',
                        help='training filenames for set B images')
    parser.add_argument('--filenamesA_val',
                        default='/media/hdd2/Datasets/ILSVRC2012/filenames/val_faceless_filenames.txt',
                        help='validation filenames for set A images')
    parser.add_argument('--filenamesB_val',
                        default='/media/hdd2/Datasets/vggface2/filenames/test_filenames_random_30k.txt',
                        help='validation filenames for set B images')
    parser.add_argument('--checkpoints_dir', default='/media/hdd2/research/privacy_pytorch/scratch', type=str, help='where to save intermediate networks')
    parser.add_argument('-j', '--num_threads', default=4, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--resume_epoch', default=None, type=int, help='continue training by loading this epoch')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', default=1, type=int, help='print frequency')
    parser.add_argument('--seed', default=1313, type=int, help='seed for initializing training (default: 1313)')
    parser.add_argument('--gpu', '--gpu_ids', default=0, type=int, help='GPU id to use (default=0)')
    parser.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='shuffle training data')
    parser.add_argument('--isTrain', dest='isTrain', action='store_true', help='Flag for training')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    parser.add_argument('--mask_size', default=128, type=int, help='mask size')
    parser.add_argument('--no_normalize_feats', dest='normalize_feats', action='store_false', help='Normalize measurements')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='simply validate')
    parser.add_argument('--train_mask', dest='train_mask', action='store_true', help='set this to allow mask weights to learn')

    parser.set_defaults(shuffle_train=True, isTrain=True, nn_mask=False, normalize_feats=True, evaluate=False, train_mask=False)

    return parser.parse_args()


def generate_parser_debug():
    parser = argparse.ArgumentParser()

    parser.dataA_dir = '/media/hdd1/Datasets/casia/two_subjects_images'
    parser.num_threads = 4
    parser.gpu = 0
    parser.epochs = 10
    parser.batch_size = 128
    parser.lr = 0.01
    parser.momentum = 0.9
    parser.weight_decay = 1e-4
    parser.print_freq = 1
    parser.shuffle_train = True
    parser.seed = 1313

    return parser
