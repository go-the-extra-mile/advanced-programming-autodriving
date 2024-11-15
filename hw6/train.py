import torch
import torch.nn as nn
import numpy as np
import random
import time
from network import ClassificationNetworkColors
from demonstration import load_demonstrations


def extract_speed(observation, batch_size):
    speed_crop = observation[84:94, 12, 0].reshape(batch_size, -1)
    speed = speed_crop.sum(dim=1, keepdim=True) / 255
    return speed


def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """

    infer_action = ClassificationNetworkColors()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    states, actions = load_demonstrations(data_folder)
    states = [torch.Tensor(state) for state in states]
    actions = [torch.Tensor(action) for action in actions]

    # make batch
    # batch: (state, class)
    # state: 96*96*3
    # class: 1
    batches = [batch for batch in zip(states, infer_action.actions_to_classes(actions))]

    # preprocessing: do not allow speed = 0 and class 6 to be in the batch
    new_batches = []
    for batch in batches:
        image = np.flip(batch[0].numpy(), 2)
        speed = extract_speed(batch[0], 1)

        # force to move forward
        if speed == 0 and batch[1].numpy()[0] == 6: # no input
            batch[1][0] = 7 # accelerate

        new_batches.append(batch)
    batches = new_batches

    # setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # temp
    print(f"Device set as: {device}")
    # pmet

    nr_epochs = 2000
    batch_size = 64
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(batches):
            # expand the batch
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            # EXPLAIN ON REPORT START
            # when the batch size is enough, train it
            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0), (-1, 96, 96, 3)) # (N, 96, 96, 3)
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1,)) # N

                # compute forward inference, compute loss
                batch_out = infer_action(batch_in) # (N, 9)
                loss = loss_function(batch_out, batch_gt) 
                # EXPLAIN ON REPORT END

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print(
            "Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs"
            % (epoch + 1, total_loss, time_left)
        )

    torch.save(infer_action, trained_network_file)
