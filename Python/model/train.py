import sys, os, json
sys.path.append('../')

from pose_estimation import PoseEstimationNetwork

import numpy as np
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from datasets.pose_dataset import PoseDataset

import argparse

# Global variables
parser = argparse.ArgumentParser(description='')
parser.add_argument('--hand', help='If you collect the left hand data, set "left", else "right".', choices=['left', 'right'], required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--earlystop', type=int, required=True)
hand = parser.parse_args().hand
epochs = parser.parse_args().epochs
earlystop = parser.parse_args().earlystop

# ref: https://qiita.com/ku_a_i/items/ba33c9ce3449da23b503
class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する


class CustomLoss(nn.Module):
    def __init__(self): 
        super(CustomLoss, self).__init__()
        self.L1loss = nn.L1Loss(reduction='sum')

    def forward(self, p, pos, r, rot):
        loss = self.L1loss(p, pos) + self.L1loss(r, rot)
        return loss

def train(epochs=1000, hand='left', sensor_size=8):
    earlystopping = EarlyStopping(patience=earlystop, verbose=True, path=f'./checkpoint_model_{hand}.pth')
    model = PoseEstimationNetwork()
    if os.path.exists(f'./checkpoint_model_{hand}.pth'):
        model.load_state_dict(torch.load(f'./checkpoint_model_{hand}.pth', map_location=torch.device('cpu')))

    loss_fnc = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) #オプティマイザはAdam
    loss_record = [] #lossの推移記録用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #デバイス(GPU or CPU)設定
    print(f'training on {device}')
    epochs = epochs #エポック数

    model.to(device) #モデルをGPU(CPU)へ

    batch_size = 16

    dataset = PoseDataset(dataset_path=f'../datasets/{hand}/dataset_{hand}_{sensor_size}.json')

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)

    for i in range(epochs):
        model.train() #学習モード

        running_loss =0.0 #記録用loss初期化
        start_time = time.time()

        for index, (x, y, pos, rot) in enumerate(train_loader):
            x = x.to(device, non_blocking=True) #シーケンシャルデータをバッチサイズ分だけGPUへ
            y = y.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            rot = rot.to(device, non_blocking=True)
            optimizer.zero_grad() #勾配を初期化
            p, r = model(x, y) # 予測
            loss = loss_fnc(p, pos, r, rot) # L = L_p + L_r
            loss.backward()  # 逆伝番
            optimizer.step()  #勾配を更新
            running_loss += loss.item()  #バッチごとのlossを足していく
        
        loss_record.append(running_loss / (index + 1)) #記録用のlistにlossを加える
        elapsed_time = time.time() - start_time
        print(f'epoch {i + 1} - loss:{running_loss / (index + 1)} - {elapsed_time} s/epoch')

        #★毎エポックearlystoppingの判定をさせる★
        earlystopping((running_loss / (index + 1)), model) #callメソッド呼び出し
        if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            break

    model = model.to('cpu')
    model.load_state_dict(torch.load(f'./checkpoint_model_{hand}.pth', map_location=torch.device('cpu')))

    torch.save(model.state_dict(), f'./model_{hand}.pth')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    size_list = []
    with open(f'../datasets/dataset_{hand}.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    for data in dataset['data']:
        if not len(data['acc']) in size_list:
            size_list.append(len(data['acc']))
    for i, size in enumerate(size_list):
        print(f'({i + 1}/{len(size_list)})')
        train(epochs=epochs, sensor_size=size, hand=hand)