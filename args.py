import argparse, time, os, json

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--act', nargs='?', type=str, default='train', help='[]')
parser.add_argument('--env', nargs='?', type=str, default='home', help='[home, floyd, work]')
# dirs
parser.add_argument('--pool_dir', nargs='?', type=str, default='./outputx/pool/', help='[]')
parser.add_argument('--q', nargs='?', type=int, default=0, help='[]')
parser.add_argument('--epochs', nargs='?', type=int, default=2000, help='[]')
parser.add_argument('--matches', nargs='?', type=int, default=10, help='[]')
parser.add_argument('--time_lim', nargs='?', type=int, default=300, help='[]')
parser.add_argument('--batch_size', nargs='?', type=int, default=16, help='[]')
parser.add_argument('--start_model', type=str, default=None)
parser.add_argument('--desc', nargs='?', type=str, default="nn_small", help='[]')
parser.add_argument('--gamma', nargs='?', type=float, default=0.99, help='')
parser.add_argument('--valid_size', nargs='?', type=int, default=100, help='[]')
parser.add_argument('--k', nargs='?', type=int, default=32, help='[]')
parser.add_argument('--lr', nargs='?', type=float, default=0.0005, help='[]')
parser.add_argument('--momentum', nargs='?', type=float, default=0.9, help='[]')
# saving + logging stuff
parser.add_argument('--pct_train', nargs='?', type=float, default=0.9, help='[]')
parser.add_argument('--chkt_every', nargs='?', type=int, default=200000, help='[]')
parser.add_argument('--log_every', nargs='?', type=int, default=20, help='[]')
parser.add_argument('--validate_every', nargs='?', type=int, default=1000, help='[]')

# zero-specific args
parser.add_argument('--num_sims', type=int, default=100, help='[]')
parser.add_argument('--verbose', type=int, default=0, help='[]')
parser.add_argument('--num_games_self', type=int, default=100, help='[]')
parser.add_argument('--num_games_eval', type=int, default=50, help='[]')
parser.add_argument('--win_thresh', type=float, default=0.55)


args = parser.parse_args()

# control Env
if args.env == 'home':
    args.data_dir = '/home/psavine/data/isolation/positions/'
    args.chkpt_dir = './outputx/checkpoints/'
    args.out_pool_dir = './outputx/out_pool/'
    args.defeated = './outputx/defeated_pool/'
    args.self_play_dir = '/home/psavine/data/isolation/selfplay/'

elif args.env == 'floyd':
    args.data_dir = '/input/positions/'
    args.chkpt_dir = '/output/checkpoints/'
    args.out_pool_dir = '/output/out_pool/'
    args.defeated = '/output/defeated_pool/'
    args.self_play_dir = '/output/selfplay/'

print(args)
print("---------------------")

# create env
for dr in [args.self_play_dir, args.chkpt_dir, args.pool_dir,args.defeated, args.out_pool_dir]:
    if not os.path.exists(dr):
        os.makedirs(dr)

start_timer = time.time()
start = '{:0.0f}'.format(start_timer)

# process booleans
args.verbose = True if args.verbose == 1 else False

if args.save != '':
    args.base_dir = '{}{}{}_{}/'.format(args.prefix, 'models/', start, args.save)
    os.mkdir(args.base_dir)
    os.mkdir(args.base_dir + 'checkpts/')
    argparse_dict = vars(args)
    with open(args.base_dir + 'params.txt', 'w') as outfile:
        json.dump(argparse_dict, outfile)
    print('Saving in folder {}'.format(args.base_dir))

