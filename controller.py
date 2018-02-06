import os
from torch import autograd
import glob
from torch.utils import data

from envs.isolation import *
from problem import loader as serial_util, dataloaders as ld
from sample_players import (RandomPlayer, open_move_score,
                            improved_score, center_score)
from agents.game_agent import (MinimaxPlayer, AlphaBetaPlayer)
from agents.nn_players import *
import time
from args import args

random.seed()
eps=1e-6

def print_game(g, w, h, o):
    print("\nWinner: {}\nOutcome: {}".format(w, o))
    print(g.to_string())
    print("Move history:\n{!s}".format(h))
    print(g.move_count)


def print_stats(moves, indices, loss):
    num_correct = torch.nonzero(indices.data  - moves.data).size(0)
    #print(num_correct)
    accuracy =  (len(moves) - num_correct)  / len(moves)
    print("accuracy: {}, validation_loss: {}, num_samples: {}".format(accuracy, loss.data[0], len(moves)))


def save_model_fmt(model, chkpt_dir, desc, start, e):
    loc = "{}model_{}_{}_{}.pkl".format(chkpt_dir, desc, start, e)
    torch.save(model, loc)
    return loc


def loadNNPlayer(loc, desc='default'):
    model = torch.load(loc)
    policy = inn.PolicyNet(model)
    return NNPlayer(policy, name=desc)


def save_game(location, serialized_game, serialized_moves, res, policy_name):
    for idx in range(len(serialized_game)):
        name = "{}{}-r-{}m{}p2.npz".format(location, policy_name, res, idx)
        np.savez_compressed(name, position=serialized_game[idx], move=serialized_moves[idx], result=res)


def print_match_stat(total_won, epoch, opp_name, game, res):
    print("wins:{}, out_of:{}, vs:{}, last_mc:{},  last_res:{}".format(total_won, epoch, opp_name, game.move_count, res))


def play_game_rl(policy, opponent_player, opp_name, args, policy_id='best', save_mode=True, verbose=False):

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=args.lr)
    start_time = time.strftime("%Y-%m-%d-%H:%M")
    total_won = 0

    for epoch in range(args.epochs):
        #
        player = NNPlayer(policy, replay=True, name=policy_id)
        game = Board(opponent_player, player)
        winner, history, o = game.play_sl()
        serilized = {'hist': history, 'duration': game.move_count, 'size': (7, 7)}
        positions_n, moves_n = serial_util.process_one_file(serilized, player=1)
        print(game.move_count, winner)

        if verbose:
            #show the game for testing
            print_game(game, winner, history, o)

        #todo clean this up
        if hasattr(winner, 'name'):
            if winner.name == policy_id:
                z_T, res = 1, 'win'
                total_won += 1
            else:
                z_T, res = -1, 'loss'
        else:
            z_T, res = -1, 'loss'

        # actions = player.saved_actions
        # assume that loss is spread around each move
        final_z = z_T / len(player.saved_actions)
        rewards = torch.from_numpy(np.repeat(final_z, len(player.saved_actions))).float().unsqueeze(-1).cuda(0)
        for action, r in zip(player.saved_actions, rewards):
            action.reinforce(r)

        optimizer.zero_grad()
        autograd.backward(player.saved_actions, [None for _ in player.saved_actions])
        optimizer.step()
        """
        #if z_T == 1 and (len(positions_n) > 0) and (len(positions_n) == len(moves_n)):
        for idx in range(game.move_count//2 - 1):
            #positions = Variable(torch.from_numpy(positions_n[idx]).float().cuda(0).unsqueeze(0))
            #moves = Variable(torch.from_numpy(moves_n[idx]).long().cuda(0).unsqueeze(0))
            #print(moves)

            logits = player.rewards[idx] #(positions)
            #print("logits", logits.size())

            action = logits.squeeze().multinomial() #.(stddev=0.2)
            #print(logits.requires_grad, logits.grad_fn)
            #print("actions",actions, actions.size(), actions.requires_grad, actions.grad_fn)

            #print(F.softmax(logits).size())
            loss = criterion(logits, action)
            #action.reinforce(torch.log(loss.data) * z_T )
            action.reinforce( z_T )
            #actions.append(action)

            #print("loss", loss.data * z_T )
            #print("loss", torch.log(loss.data) * z_T )
            optimizer.zero_grad()
            autograd.backward(action, [None for _ in action], retain_graph=True) #

            optimizer.step()
            positions = Variable(torch.from_numpy(positions_n[idx]).float().cuda(0).unsqueeze(0))
                moves = Variable(torch.from_numpy(moves_n[4]).long().cuda(0).unsqueeze(0))
                #print(moves)

                logits = policy(positions)
                print("logits", logits.size())

                actions = logits.squeeze().normal(stddev=0.2)
                print(logits.requires_grad, logits.grad_fn)
                print("actions",actions, actions.size(), actions.requires_grad, actions.grad_fn)
                print(F.softmax(logits).size())
                loss = criterion(logits, moves.squeeze() )
                actions.reinforce(loss.data * z_T )

                print("loss", loss.data)
                positions = Variable(torch.from_numpy(positions_n).float().cuda(0))
                moves = Variable(torch.from_numpy(moves_n).long().cuda(0))
                logits = policy(positions)
                print("logits", logits.size(), logits.requires_grad, logits.grad_fn)

                rewards, _ = logits.squeeze().max(-1)
                actions = logits.squeeze().multinomial()

                print("rewards",  rewards.data.unsqueeze(-1))
                print("actions",  actions.size(), actions.requires_grad, actions.grad_fn)
                #print(F.softmax(logits).size())

                #loss = criterion(logits, moves.squeeze())
                #print("loss", loss, loss.size())
                #zs = (rewards - rewards.mean()) / (rewards.std() + eps) * -z_T
                #zs = -z_T * torch.log(torch.nn.utils.clip_grad_norm(logits, eps, 1.0 - eps))
                #print("normed", zs)

                print("-----------------------------------------------------")
                #actions.reinforce(zs.data.unsqueeze(-1)  )

                outcomes = policy.saved_actions #saved logits
                policy_states = policy.rewards
                R = 0
                rewards = []
                actions = []

                for act in policy_states:
                    actions.append(act.multinomial())

                for r in policy.rewards[::-1]:
                    R = r + game.move_count * R
                    rewards.insert(0, R)
         """

        del player.rewards[:]
        del player.saved_actions[:]

        #save game to selfplay
        if save_mode:
            save_game(args.self_play_dir, positions_n, moves_n, z_T, policy_id)

        #report
        if epoch % 10 == 0 and epoch != 0:
            print_match_stat(total_won, epoch, opp_name, game, res)

    return policy, total_won/args.epochs


def play_game_questionable(policy, opponent_player, opp_name, args, policy_id='best', save_mode=True, verbose=False):

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=args.lr)
    start_time = time.strftime("%Y-%m-%d-%H:%M")
    total_won = 0

    for epoch in range(args.epochs):
        #for match in range(args.matches):
        player = NNPlayer(policy, name=policy_id)
        game = Board(opponent_player, player)
        winner, history, o = game.play_sl()
        serilized = {'hist': history, 'duration': game.move_count, 'size': (7, 7)}
        positions_n, moves_n = serial_util.process_one_file(serilized, player=1)

        if verbose:
            print_game(game, winner, history, o)
        if hasattr(winner, 'name'):
            if winner.name == policy_id:
                z_T, res = 1, 'win'
                total_won += 1
            else:
                z_T, res = -1, 'loss'
        else:
            z_T, res = -1, 'loss'

        #ITS Reinforcement...POsitive REinforcment learning.. sooo positive!
        if z_T == 1 and (len(positions_n) > 0) and (len(positions_n) == len(moves_n)):
            positions = Variable(torch.from_numpy(positions_n).float().cuda(0))
            moves = Variable(torch.from_numpy(moves_n).long().cuda(0))

            optimizer.zero_grad()
            logits = policy(positions)
            loss = criterion(logits, moves.squeeze(-1))
            _, indices = logits.max(-1)

            loss.backward()
            optimizer.step()
            ld = loss.data[0]
        else:
            ld = 'none'

        del player.rewards[:]
        del player.saved_actions[:]

        if save_mode:
            save_game(args.self_play_dir, positions_n, moves_n, z_T, policy_id)

        if (epoch + 1) % 10 == 0:
            print_match_stat(total_won, epoch, opp_name, game, res)

    return policy, total_won/args.epochs



# floyd run --env pytorch-0.2 --gpu "python controller.py --act reinforce  --env floyd --epochs 200 --start_model model_nn_small_0.006865953999977137_2.pkl --matches 10000"
def tournament_schedule(args):
    "args: epochs, matches, self_play_dir start_model, pool_dir, out_pool_dir"

    base_model = args.pool_dir + args.start_model
    policy = torch.load(base_model)
    modelglob = glob.glob(args.pool_dir + '*')

    inactive = [["MM_Improved", MinimaxPlayer(score_fn=improved_score)],
                ["AB_Open", AlphaBetaPlayer(score_fn=open_move_score)],
                ["AB_Center", AlphaBetaPlayer(score_fn=open_move_score)],
                ["AB_Improved", AlphaBetaPlayer(score_fn=improved_score)]]

    agents = {"Random": RandomPlayer(),
              "Random2": RandomPlayer(),
              "MM_Open": MinimaxPlayer(score_fn=open_move_score),
              "MM_center": MinimaxPlayer(score_fn=center_score)}

    def print_round(win_pct, match, opp_name):
        print("win_pct:{}, match:{}, vs:{} ---- base:{}".format(win_pct, match, opp_name, args.start_model))

    defeated = []
    models = modelglob + list(agents.keys())
    best_hist = []
    start_time = time.strftime("%Y-%m-%d-%H:%M")

    if args.q == 0:
        game_fn = play_game_rl
    else:
        game_fn = play_game_questionable

    for match in range(args.matches):

        # load an Opponent
        opp_model_loc = random.choice(models)
        if opp_model_loc in agents:
            opp_name = opp_model_loc
            opponent_player = agents[opp_model_loc]
        else:
            opp_name = os.path.basename(opp_model_loc)
            opponent_player = loadNNPlayer(opp_model_loc)

        # play game
        policy, win_pct = game_fn(policy, opponent_player, opp_name, args, policy_id='best_in')
        print_round(win_pct, match, opp_name)

        if win_pct >= 0.99:
            print("---------------")
            print("{} has been defeated at match {}".format(opp_name, match))
            try:
                if opp_model_loc in agents:
                    if len(inactive) > 0:
                        print("new Search opponent has been created: {}".format(inactive[0][0]))
                        models.append(inactive[0][0])
                        agents[inactive[0][0]] = inactive[0][1]
                        del inactive[0]
                else:
                    add_pool = glob.glob(args.out_pool_dir + '*')
                    for nn in add_pool:
                        if nn not in defeated:
                            models.append(nn)
                            print("new NN opponent has been created: {}".format(nn))
                            break

                models.remove(opp_model_loc)
                defeated.append(opp_model_loc)
            except:
                print("you are an idiot. remember to write a unittest. fuck state")

        if (match + 1) % 50 == 0:
            policy_loc = save_model_fmt(policy, args.out_pool_dir, start_time, '-top-', match)

        best_hist.append([opp_name, win_pct])

    print("---------------Tournament complete------------")
    print("---------------defeated opponents--------------")
    # (print(x) for x in defeated)
    print("---------------FINAL RESULTS------------------")
    # running_total = 0
    for m in best_hist:
        print("win_pct:{}, vs: {}".format(m[1], m[0]))


def play_with_friend(args, checkpoint, checkpoint2=None):
    # intialize ρ = σ
    policy = load_model_fmt(checkpoint)

    # policy = inn.Net()
    if not checkpoint2:
        player1 = AlphaBetaPlayer(score_fn=improved_score)
    else:
        # todo replace with previous verion
        player1 = AlphaBetaPlayer(score_fn=improved_score)

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(policy.parameters(), lr=args.lr)
    win, loss = 1, -1
    wins = 0
    results = []
    for i in range(args.epochs):
        # white = np.random.randint(0, high=1)

        player2 = NNPlayer(policy)
        game = Board(player1, player2)

        start = timeit.timeit()
        winner, _, outcome = game.play_sl(time_limit=args.time_lim, sl_player=2)
        end = timeit.timeit()
        print("game duration: {}".format(1000* (end - start)))

        player_SL = game.get_player_N(2)
        policy = player_SL.model
        rewards = player_SL.rewards * z_T

        if game.is_winner(player2):
            z_T = win
            wins += 1
        else:
            z_T = loss

        print(z_T, winner)

        for action, r in zip(player_SL.saved_actions, rewards):
            action.reinforce(r)

        optimizer.zero_grad()
        autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
        optimizer.step()

        del policy.rewards[:]
        del policy.saved_actions[:]
        if (i + 1) % 10 == 0:
            print("num wins {}".format(wins, ))
            wins = 0


def load_model_fmt(chkpt_dir, model=None):
    if not model:
        f = os.listdir(chkpt_dir)[:-1]
        loc = chkpt_dir + f[0]
    else:
        loc = chkpt_dir + f[0]

    print(loc)
    model = torch.load(loc)
    return model


def run_validation(model, valid_loader, valid_size, criterion):
    """
        Pick batch from validation set and run it. THese be shuffled
    """
    positions, moves = valid_loader.__iter__().__next__()
    positions = Variable(positions.cuda(0))
    moves = Variable(moves.cuda(0))

    logits = model(positions)
    # indices = logits.max(-1)
    loss = criterion(logits, moves)

    # num_correct = torch.nonzero(moves.data - indices.data).size(0)
    # accuracy =  (valid_size - num_correct)  / valid_size
    print("validation_loss: {}, num_samples: {}".format(loss.data[0], valid_size))
    # print("accuracy: {}, validation_loss: {}, num_samples: {}".format(loss.data[0], valid_size))


def init_value_net(policy_loc):
    model = torch.load(policy_loc)
    policy = inn.PolicyNet(model).cuda(0)
    return policy

def train_supervised(args, model, criterion):
    """
        size_average (bool, optional) – By default, the losses are averaged
        over observations for minibatch.
        However, if the field size_average is set to False, the losses are
        instead summed for minibatch.
        data_dir,epochs=50 lr=0.0002, batch_size=16, valid_size=100
    """
    # alphago does not use momemntum, but a few people said it trained faster.
    # also adadelta seams to be mode in style now...
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # dataloaders - train set is majority of set
    # validation is some 5% of set. I can augment if nec. also shuffled
    train_data = ld.CoupledLoader(args.data_dir, train=True, vl=args.vl, pct_train=args.pct_train)
    valid_data = ld.CoupledLoader(args.data_dir, train=False, vl=args.vl, pct_train=args.pct_train)

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=args.shuffle)
    valid_loader = data.DataLoader(valid_data, batch_size=args.valid_size, num_workers=1, shuffle=True)
    # m=16 deepmind
    step = 0
    for epoch in range(args.epochs):
        for i, (positions, targets) in enumerate(train_loader):
            step +=1
            positions = Variable(positions.cuda(0))
            targets = Variable(targets.cuda(0))

            optimizer.zero_grad()
            logits = model(positions)
            # print("pos", logits.size(), targets.size())
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            if (i+1) % args.log_every == 0:
                print("Epoch: {}/{}, step: {}, training_loss: {:.5f}".format(
                    epoch + 1, args.epochs, i + 1, loss.data[0]))

            if i % args.validate_every == 0:
                run_validation(model, valid_loader, args.valid_size, criterion)

            # save atleast once per epoch
            if step % args.chkt_every == 0:
                torch.save(model, "{}model_{}_e{}_s{}.pkl".format(args.chkpt_dir, args.desc,epoch, step))
    # Final save
    torch.save(model, "{}model_{}_e{}_s{}.pkl".format(args.chkpt_dir, args.desc, "-Final", step))


def setup_value_training(args):
    criterion = nn.MSELoss().cuda(0)
    args.data_dir = '/home/psavine/data/isolation/selfplay2/'
    args.shuffle = True
    args.vl = True
    model = torch.load(args.start_model)
    value = inn.PolicyNet(model).cuda(0)
    print(value)
    train_supervised(args, value, criterion)


def setup_policy_training(args):
    #initalize model #NLL in pytorch ~ categorical_crossentropy in tf
    criterion = nn.CrossEntropyLoss().cuda(0)
    args.gamedir = None
    args.shuffle = False
    if args.start_model is not None:
        model = torch.load(args.start_model)
    else:
        model = inn.Net(k=args.k).cuda(0)
    train_supervised(args, model, criterion)


def doinference(args):
    criterion = nn.MSELoss().cuda(0)
    args.data_dir = '/home/psavine/data/isolation/games/'
    model = '200-Final.pkl'
    model = torch.load(args.start_model)
    train_data = ld.CoupledLoader(args.data_dir, train=True, vl=args.vl, pct_train=args.pct_train)
    train_loader = data.DataLoader(train_data, batch_size=1, num_workers=2, shuffle=True)



# floyd run --env pytorch-0.2 --gpu "python controller.py --act reinforce  --env floyd --epochs 200 --start_model model_nn_small_0.006865953999977137_2.pkl --matches 1000"
# python controller.py --act value --start_model ./outputx/checkpoints/model_nn_test_0.008578838998801075_2.pkl
# 200-Final.pkl
if __name__ == "__main__":
    # Functions
    if args.act == 'policy':
        setup_policy_training(args)

    elif args.act == 'value':
        setup_value_training(args)

    elif args.act == 'reinforce':
        tournament_schedule(args)

    elif args.act == 'log':
        log_graph('./misc/training')

    elif args.act == 'testload':
        load_model_fmt(args.chkpt_dir)

    else:
        args.gamma = 1
        print(args.gamma, args.pool_dir)
        print("NYI")

#python controller.py --act train --k 16 --lr 0.001 --momentum 0.0

#floyd run --env pytorch-0.2 --gpu --data psavine/isolation/1 "python controller.py --act train --epochs 50 --env floyd"

#floyd run --env pytorch-0.2 --gpu --data psavine/isolation/1 "python controller.py --act train --epochs 50 --env floyd"





#sanity check code reference

# https://github.com/maxpumperla/betago/blob/master/betago/training/checkpoint.py
# opt = Adadelta(clipnorm=0.25)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#https://github.com/TheDuck314/go-NN/blob/master/engine/MoveTraining.py
#def loss_func(logits):
#    move_indices = tf.placeholder(tf.int64, shape=[None])
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, move_indices)
#    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

#    correct_prediction = tf.equal(tf.argmax(logits,1), move_indices)
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    return move_indices, cross_entropy_mean, accuracy


#for model, hist in results.items():
    #running_total = 0
    #for opp, res in hist:
        #running_total += res
    #print("model {}, win_pct:{}, num".format(model, running_total / max(1, len(hist))))