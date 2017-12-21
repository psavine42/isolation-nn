data_keys = ['hist', 'duration', 'result', 'size']
defkeys = ['hist', 'size', 'duration']

def main():
    ld.normalize_dataset('./data/', './positions/')


def test_make_file(idx):
    dt = ld.read_npz(data_dir + os.listdir(data_dir)[idx], data_keys)
    print(dt)
    return ld.process_one_file(dt)


def testnet():
    model = inn.Net()

    batch, moves = test_make_file()
    b1 = batch[5]
    # print(b1)
    print(b1.shape)
    n1 = torch.from_numpy(batch[0]).float().unsqueeze(0)
    # print(n1.size())
    v = Variable(n1)
    logits, outputs = model(v)


def testload():

    model = inn.Net().cuda(0)
    batch_s = 2

    criterion = nn.CrossEntropyLoss().cuda(0)
    trainloader = data.DataLoader(loader, batch_size=batch_s, num_workers=2, shuffle=False)

    v_total = 0
    v_correct = 0
    for i, (positions, moves) in enumerate(trainloader):
        if i < 10:
            positions = Variable(positions.cuda(0))
            moves = Variable(moves.squeeze().cuda(0))

            #print(moves.squeeze().size())
            v_total += 1
            #print("pos", positions.size())
            logits = model(positions)
            values, indices = logits.max(-1)
            loss = criterion(logits, moves)

            correct = torch.nonzero(moves.data - indices.data).size(0)
            accuracy =  (moves.data.size(0) - correct)  / moves.data.size(0)
            print("accuracy: {}, loss: {}, num-samples: {}".format(accuracy, loss.data[0], moves.data.size(0)))