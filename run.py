from args import args
from envs.env import IsolationEnvSeq

if __name__ == '__main__':
    if args.act == 'testrun':
        main = IsolationEnvSeq(args)
        main.forward_sync()
