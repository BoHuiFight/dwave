import os
import zipfile
import numpy as np
import torch


def load_data(pathNum,pathLen):

    features=np.load("../PCN/data/chengdu/features.npy")

    roadids = []

    districtids = []
    with open("data/chengdu/city_district.txt") as fr:
        for l in fr:
            temp = l.split("\t")
            id = temp[0]
            districtids.append(id)

    dic = {}
    with open("data/chengdu/boundary_en_chengdu.txt") as fr:
        for l in fr:
            temp = l.split("\t")
            id = temp[0]
            if id not in districtids:
                roadids.append(id)

    col_mean = np.true_divide(features.sum(0), (features != 0).sum(0))
    col_max=features.max(0)
    zeroInds = np.where(~np.isnan(col_mean))[0]
    inds = np.where(features == 0)
    features[inds] = np.take(col_max, inds[1])
    features = features[:, zeroInds]
    roadids=np.array(roadids)
    trajSamples=[]
    with open("data/chengdu/s2osm-traj-cd.txt") as fr:
        for l in fr:
            temp = l.rstrip('\n').split(",")
            indices = [(i, i + pathLen) for i in range(len(temp) - (pathLen+ 1))]
            for i, j in indices:
                trajSamples.append(temp[i: i + pathLen])
    trajSamples=np.array(trajSamples)
    lnodes=trajSamples[:,-1]
    lnodes=np.unique(np.array(lnodes))

    r2sDic={}
    s2rDic={}
    fixSet=set(lnodes)


    X=[]
    ind=0
    print(features.shape)
    with open("data/chengdu/s2osm-st-cd.txt") as fr:
        for l in fr:
            temp = l.rstrip('\n').split("\t")
            instersect=set(temp[1].split(",")).intersection(fixSet)
            instersect=list(instersect)
            if len(instersect)>0:
                for t in instersect:
                  if t not in s2rDic:
                    s2rDic[t]=ind
                r2sDic[ind]=instersect
                X.append(features[:,roadids.index(temp[0])])
                ind+=1

    keys=[*s2rDic]
    newS2Rdic={}
    newR2SDic={}

    for i in range(len(keys)):
      newS2Rdic[i]=s2rDic[keys[i]]
    s2rDic=newS2Rdic  
    curIndex=len(s2rDic)
    for road in r2sDic:
      intersections=r2sDic[road]
      temp=[keys.index(ist) for ist in intersections if s2rDic[keys.index(ist)]==road]
      if len(temp)>0:
        newR2SDic[road]=temp
      else:
        newR2SDic[road]=[curIndex]
        s2rDic[curIndex]=road
        curIndex+=1
    r2sDic=newR2SDic

    allTrajs=np.load("../PCN/traj-cd-55.npy",allow_pickle=True)
    trajDic={}
    for traj in allTrajs:
      if traj[-1] not  in trajDic:
        trajDic[traj[-1]]=[traj]
      else:
        trajDic[traj[-1]].append(traj)
    for traj1 in trajDic:
      trajDic[traj1]=np.array(trajDic[traj1]).astype(int)


    X=np.array(X)
    X=X.transpose((1, 0))
    X=np.expand_dims(X, axis=2)
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    return X,r2sDic,s2rDic,trajDic,keys

def jcSimilarity(interval1, interval2):
    if interval1.begin>interval2.end or interval2.begin>interval1.end:
      return 0
    else:
      return (min(interval1.end,interval2.end)-max(interval1.begin,interval2.begin))/(max(interval1.end,interval2.end)-min(interval1.begin,interval2.begin))

def random_trajs(treeDic,r2sDic,s2rDic,keys,pathNum,tIndex):
    ind=0
    trajsDic={}

    randomtrajs={}
    for key in keys:
        tempTree=treeDic[key]
        start=random.randint(1, 44) 
        trajSamples=sorted(tempTree[tIndex:tIndex+6])

        jcValues=[]
        for traj in trajSamples:
          jcValues.append(jcSimilarity(traj,Interval(start, start+6, (start, start+6))))

        sortIndex=(-np.array(jcValues)).argsort()[:pathNum]

        for ind in sortIndex:
          temp1=trajSamples[ind]
          tempTraj=[]
          for seg in temp1.data:
            if str(seg) in keys:
              tempTraj.append(int(keys.index(str(seg))))
          if len(tempTraj)<5:
            tempTraj.extend([tempTraj[-1]]*(5-len(tempTraj)))
          if tempTraj[-1] not in randomtrajs:
            randomtrajs[tempTraj[-1]]=[tempTraj]
          else:
            if len(randomtrajs[tempTraj[-1]]) <5:
              randomtrajs[tempTraj[-1]].append(tempTraj)
        ind+=1
    for s in s2rDic:
      if s not in  randomtrajs:
        randomtrajs[s]=[[s]*5]* pathNum

    return randomtrajs


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

