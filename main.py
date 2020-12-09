import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dwave import PCN
from utils import generate_dataset,load_data,masked_mae_loss
import utils

use_gpu = False


parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',default=True,
                    help='Enable CUDA')
parser.add_argument('--model', type=str, default='PCN',
                    help='select model')
parser.add_argument('--num_timesteps_input', type=int, default=6,
                    help='input slices')
parser.add_argument('--num_timesteps_output', type=int, default=6,
                    help='output slides')
parser.add_argument('--epochs', type=int, default=50,
                    help='configure epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='select model')
parser.add_argument('--pathNum', type=int, default=5,
                    help='select model')
parser.add_argument('--pathLen', type=int, default=5,
                    help='select model')
parser.add_argument('--lr', type=int, default=1e-1,
                    help='select model')

args = parser.parse_args()
args.device = None
num_timesteps_input=args.num_timesteps_input
num_timesteps_output=args.num_timesteps_output
epochs=args.epochs
batch_size=args.batch_size
pathNum=args.pathNum
pathLen=args.pathLen


if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:0')
else:
    args.device = torch.device('cpu')


def train_epoch(training_input, training_target, batch_size,randomtrajs):

    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()


        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]

        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        #print(os.times().elapsed)
        out = net(X_batch,r2sDic,s2rDic,randomtrajs)
        loss = masked_mae_loss(out, y_batch)
        print(loss)
        loss.backward()
        
        optimizer.step()
        #print(os.times().elapsed)
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

if __name__ == '__main__':
    torch.manual_seed(1)

    X,r2sDic,s2rDic,trajDic,keys = load_data(pathNum,pathLen)
    split_line1 = int(X.shape[2] * 0.7)
    split_line2 = int(X.shape[2] * 0.8)
    split_line3 = int(X.shape[2])

    means = np.mean(X[:, :, :split_line1], axis=(0, 2))
    stds = np.std(X[:, :, :split_line1], axis=(0, 2))
    X = X - means[0]
    X = X / stds[0]
    
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)


    if args.model=='PCN':
        net = PCN(len(s2rDic),
                    training_input.shape[3],
                    num_timesteps_input,
                    num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(epochs):

        randomtrajs=utils.random_trajs(trajDic,s2rDic,keys,pathNum)
        loss = train_epoch(training_input, training_target,batch_size,randomtrajs)
        training_losses.append(loss)
        print("epoch:"+str(epoch))
        print("Training loss: {}".format(training_losses[-1]))

        #Run validation

        val_losses = []
        val_mae = []
        with torch.no_grad():
            batch_loss=np.zeros(((split_line3-split_line2)//batch_size+1,6))
            batch_maes=np.zeros(((split_line3-split_line2)//batch_size+1,6))
            batch_mape=np.zeros(((split_line3-split_line2)//batch_size+1,6))
            batch_rmse=np.zeros(((split_line3-split_line2)//batch_size+1,6))
            for i in range((split_line3-split_line2)//batch_size+1):
              net.eval()
              mini_test_input = test_input[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
              mini_test_target = test_target[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
              out = net(mini_test_input,r2sDic,s2rDic,randomtrajs)
              for j in range(out.shape[2]):
                test_loss = loss_criterion(out[:,:,j], mini_test_target[:,:,j]).to(device="cpu")
                val_losses.append(np.asscalar(test_loss.detach().numpy()))
                out_unnormalized = out[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
                target_unnormalized = mini_test_target[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
                mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                batch_rmse[i,j]=utils.masked_rmse_np(out_unnormalized, target_unnormalized)
                batch_mape[i,j]=utils.masked_mape_np(target_unnormalized,out_unnormalized)
                batch_maes[i,j]=mae
            # print(batch_loss)
            # print(batch_maes)
            print("val loss: {}".format(batch_loss.mean(axis=0)))
            print("val MAE: {}".format(batch_maes.mean(axis=0)))
            print("val MAPE: {}".format(batch_mape.mean(axis=0)))
            print("val RMSE: {}".format(batch_rmse.mean(axis=0)))
    # Run test
    test_losses = []
    test_maes = []
    with torch.no_grad():
        randomtrajs=utils.random_trajs(trajDic,s2rDic,keys,pathNum)
        for i in range((split_line3-split_line2)//batch_size+1):
          net.eval()
          mini_test_input = test_input[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
          mini_test_target = test_target[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
          mask1=torch.from_numpy(mask).to(device=args.device)
          out = net(mini_test_input,r2sDic,s2rDic,randomtrajs)
          for j in range(out.shape[2]):
            test_loss = loss_criterion(out[:,:,j], mini_test_target[:,:,j]).to(device="cpu")
            print(test_loss)
            out_unnormalized = out[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
            target_unnormalized = mini_test_target[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            test_maes.append(mae)
        print("test MAE: {}".format(batch_maes.mean(axis=0)))

