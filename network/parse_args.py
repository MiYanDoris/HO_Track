from utils import boolean_string


def add_args(parser):
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False, help='use model.visualize() (show plt figures)')
    parser.add_argument('--debug_save', action='store_true', default=False, help='use model.visualize() (save plt figures)')
    parser.add_argument('--save', action='store_true', default=False, help='save tracking results for generating .gif file')
    parser.add_argument('--track', default=None, choices=['hand', 'obj', 'both', 'hand_IKNet', 'obj_opt', False], help='track!')

    parser.add_argument('--obj_config', type=str, default=None)
    parser.add_argument('--obj_category', type=str, default=None)
    parser.add_argument('--experiment_dir', type=str, default=None)

    parser.add_argument('--coord_exp/dir', type=str, default=None)
    parser.add_argument('--coord_exp/resume_epoch', type=int, default=None)

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--cuda_id', type=int, default=None)
    parser.add_argument('--total_epoch',  default=None, type=int)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--lr_policy', type=str, default=None)
    parser.add_argument('--lr_gamma', type=float, default=None)
    parser.add_argument('--lr_step_size', type=int, default=None)
    parser.add_argument('--lr_clip', type=float, default=None)

    parser.add_argument('--num_workers', type=int,  default=4, help='num_workers in data_loader')

    parser.add_argument('--num_points', type=int,  default=None)

    parser.add_argument('--freq/save', type=int,  default=None, help='ckpt saving frequency in epochs')

    parser.add_argument('--pointnet_cfg/camera', type=str, default=None)

    parser.add_argument('--network/type', type=str, default=None)
    parser.add_argument('--network/backbone_out_dim', type=int, default=None)


    return parser

