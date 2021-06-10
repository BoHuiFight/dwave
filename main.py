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


parser = argparse.ArgumentParser(description='Twave')
parser.add_argument('--enable-cuda', action='store_true',default=True,
                    help='Enable CUDA')
parser.add_argument('--model', type=str, default='T-wave',
                    help='select model')
parser.add_argument('--num_timesteps_input', type=int, default=6,
                    help='input slices')
parser.add_argument('--num_timesteps_output', type=int, default=6,
                    help='output slides')
parser.add_argument('--epochs', type=int, default=50,
                    help='configure epochs')
parser.add_argument('--batch_size', type=int, default=1,
                    help='select model')
parser.add_argument('--pathNum', type=int, default=7,
                    help='select model')
parser.add_argument('--pathLen', type=int, default=7,
                    help='select model')
parser.add_argument('--lr', type=int, default=1e-2,
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


def train_epoch(batch_size):

    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()
        randomtrajs=utils.random_trajs(trajDic,r2sDic,s2rDic,keys,pathNum,i%144)

        indices = permutation[i:i + batch_size]
        X_batch,X_batch_daily,X_batch_weekly,X_batch_coarse,  y_batch = training_input[indices], training_daily_input[indices],training_weekly_input[indices],training_coarse_input[indices],training_target[indices]
        X_batch = X_batch.to(device=args.device)
        X_batch_daily = X_batch_daily.to(device=args.device)
        X_batch_weekly = X_batch_weekly.to(device=args.device)
        X_batch_coarse = X_batch_coarse.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        out = net(X_batch,X_batch_daily,X_batch_weekly,X_batch_coarse,r2sDic,s2rDic,randomtrajs)
        loss = masked_mae_loss(out, y_batch)
        loss.backward()
        
        optimizer.step()
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
    val_original_data = X[:, :, split_line1-1008:split_line2]
    test_original_data = X[:, :, split_line2-1008:]

    training_input, training_daily_input,training_weekly_input,training_coarse_input,training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input,val_daily_input, val_weekly_input, val_coarse_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input,test_daily_input,test_weekly_input, test_coarse_input,test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    print(training_input.shape)
    print(training_target.shape)
    if args.model=='T-wave':
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
        loss = train_epoch(batch_size)
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

            for i in range((split_line3-split_line2)//batch_size-1):
              net.eval()
              randomtrajs=utils.random_trajs(trajDic,r2sDic,s2rDic,keys,pathNum,(i+93)%144)
              mini_test_input = test_input[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
              mini_test_input_daily = test_daily_input[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
              mini_test_input_weekly = test_weekly_input[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
              mini_test_input_coarse = test_coarse_input[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)
              mini_test_target = test_target[i*batch_size:min((i+1)*batch_size,split_line3)].to(device=args.device)

              out = net(mini_test_input,mini_test_input_daily,mini_test_input_weekly,mini_test_input_coarse,r2sDic,s2rDic,randomtrajs)
              for j in range(out.shape[2]):
                test_loss = loss_criterion(out[:,:,j], mini_test_target[:,:,j]).to(device="cpu")
                val_losses.append(np.asscalar(test_loss.detach().numpy()))
                out_unnormalized = out[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
                target_unnormalized = mini_test_target[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
                mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                batch_rmse[i,j]=utils.masked_rmse_np(out_unnormalized, target_unnormalized)
                batch_mape[i,j]=utils.masked_mape_np(target_unnormalized,out_unnormalized)
                batch_maes[i,j]=mae

            print("val loss: {}".format(batch_loss.mean(axis=0)))
            print("val MAE: {}".format(batch_maes.mean(axis=0)))
            print("val MAPE: {}".format(batch_mape.mean(axis=0)))
            print("val RMSE: {}".format(batch_rmse.mean(axis=0)))
    # Run test
    test_losses = []
    test_maes = []
    with torch.no_grad():
        for i in range((split_line2-split_line1)//batch_size+1):
          net.eval()
          mini_test_input = test_input[i*batch_size:min((i+1)*batch_size,split_line2)].to(device=args.device)
          mini_test_input_daily = test_daily_input[i * batch_size:min((i + 1) * batch_size, split_line2)].to(
              device=args.device)
          mini_test_input_weekly = test_weekly_input[i * batch_size:min((i + 1) * batch_size, split_line2)].to(
              device=args.device)
          mini_test_target = test_target[i*batch_size:min((i+1)*batch_size,split_line2)].to(device=args.device)
          randomtrajs = utils.random_trajs(trajDic, r2sDic, s2rDic, keys, pathNum, (i + 89) % 144)
          out = net(mini_test_input,mini_test_input_daily,mini_test_input_weekly,mini_test_input_coarse,r2sDic,s2rDic,randomtrajs)
          for j in range(out.shape[2]):
            test_loss = loss_criterion(out[:,:,j], mini_test_target[:,:,j]).to(device="cpu")
            print(test_loss)
            out_unnormalized = out[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
            target_unnormalized = mini_test_target[:,:,j].detach().cpu().numpy()*stds[0]+means[0]
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            test_maes.append(mae)
        print("test MAE: {}".format(batch_maes.mean(axis=0)))


