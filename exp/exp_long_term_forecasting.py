from datetime import datetime
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.evaluate_classification import evaluate_classification
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils.losses import mape_loss, mase_loss, smape_loss, VAELoss,ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
import webbrowser
import subprocess

warnings.filterwarnings('ignore')

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def contrastive_loss(z_anchor, z_pos, z_neg, margin=1.0):
    d_pos = F.pairwise_distance(z_anchor, z_pos)
    d_neg = F.pairwise_distance(z_anchor, z_neg)
    return torch.mean(F.relu(d_pos - d_neg + margin))


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        current_time = datetime.now().strftime("%m%d_%H%M")
        log_dir = f'runs/{args.model}_camera_board-{current_time}'
        self.writer = SummaryWriter(log_dir)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = self.args.loss
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
        elif loss_name == 'CrossEntropyLoss':
            # weights = torch.tensor([0.266919,0.182162, 0.166005, 0.137599,0.088173,0.159142], dtype=torch.float32).to(self.device)
            # return nn.CrossEntropyLoss(weight=weights) 
            return nn.CrossEntropyLoss()
        elif loss_name == 'saitoVAE':
            return VAELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        criterion =  nn.CrossEntropyLoss() 
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                if not self.args.camera:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs[:, -self.args.pred_len: ].to(self.device)
                    batch_y_reshape = batch_y.reshape(-1, batch_y.size(2))
                    outputs_reshape = outputs.reshape(-1, outputs.size(2))
                    batch_y_reshape = batch_y_reshape.squeeze(1) 
                    outputs_reshape = outputs_reshape.float()
                    batch_y_reshape = batch_y_reshape.long()
                    loss = criterion(outputs_reshape, batch_y_reshape) # classes 0-5 for camera
                total_loss.append(loss)
        total_loss = [loss.cpu() for loss in total_loss]  # 将所有损失移动到CPU
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # print(f"Epoch: {epoch + 1}, Step: {i + 1}/{train_steps}")
                # print(f"Batch size: {batch_x.size(0)}, Input shape: {batch_x.shape}, Target shape: {batch_y.shape}")
                # print(f"Batch x mark shape: {batch_x_mark.shape}, Batch y mark shape: {batch_y_mark.shape}")
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    if not self.args.camera:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                    else:
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        outputs = outputs[:, -self.args.pred_len: ].to(self.device)
                        batch_y_reshape = batch_y.reshape(-1, batch_y.size(2))
                        outputs_reshape = outputs.reshape(-1, outputs.size(2))
                        batch_y_reshape = batch_y_reshape.squeeze(1) 
                        outputs_reshape = outputs_reshape.float()
                        batch_y_reshape = batch_y_reshape.long()
                        # print(outputs_reshape.shape,batch_y_reshape.shape)
                        loss = criterion(outputs_reshape, batch_y_reshape) # camera - 1 for 0-5 class
                        
                        # print(outputs_reshape, batch_y_reshape)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward() 
                    if (i + 1) % 400 == 0:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:  # 仅在梯度非 None 时记录
                                self.writer.add_histogram(f'{name}.grad', param.grad, epoch)
                    model_optim.step()
            current_lr = model_optim.param_groups[0]['lr']
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            print("Epoch: {} cost time: {} LearningRate:{}".format(epoch + 1, time.time() - epoch_time,current_lr))
            train_loss = np.average(train_loss)
            self.writer.add_scalar('Loss/train_loss', train_loss, epoch)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            self.writer.add_scalar('Loss/vali_loss', vali_loss, epoch)

            test_loss = self.vali(test_data, test_loader, criterion)
            self.writer.add_scalar('Loss/test_loss', test_loss, epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)


            if early_stopping.early_stop:
                print("Early stopping")
                break
            counter = early_stopping.counter
            adjust_learning_rate(model_optim, epoch + 1,self.args, counter)

        # 关闭 writer
        self.writer.close()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        # 在循环前统计总耗时和总帧数
        total_infer_time = 0
        total_frames = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                start_time = time.perf_counter()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # end timing
                end_time = time.perf_counter()
                infer_time = end_time - start_time
                total_infer_time += infer_time

                # update total frames: pred_len * batch_size
                total_frames += batch_x.shape[0] * self.args.pred_len

                f_dim = -1 if self.args.features == 'MS' else 0
                if not self.args.camera:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs[:, -self.args.pred_len: ].to(self.device)
                    # batch_y = batch_y - 1
                    outputs = outputs.argmax(-1).cpu().numpy()
                    batch_y = batch_y.squeeze(-1).cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
      

                if not self.args.camera:
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                else :
                    preds.extend(outputs)
                    trues.extend(batch_y)
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], trues[:][-1]), axis=None)
                        pd = np.concatenate((input[0, :, -1], preds[:][-1]),axis=None)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        if not self.args.camera:       
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if self.args.camera:
            avg_time_per_frame_ms = (total_infer_time / total_frames) * 1000
            pfs = total_frames / total_infer_time

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999            
        if self.args.camera:
            preds = np.concatenate(preds).astype(int)
            trues = np.concatenate(trues).astype(int)
            print('test shape:', len(preds), len(trues))
            accuracy,precision,recall,f1,roc_auc = evaluate_classification(preds, trues)
            print('accuracy:{}, precision:{}, recall:{},f1:{},roc_auc:{}'.format(accuracy,precision,recall,f1,roc_auc))
            f = open(f"{self.args.f_name}.txt", 'a')
            f.write(setting + "  \n")
            f.write('accuracy:{}, precision:{}, recall:{},f1:{},roc_auc:{}'.format(accuracy,precision,recall,f1,roc_auc))
            f.write('\n')
            if self.args.camera:
                f.write('\n🚀 平均推理时间: {:.2f} ms/frame, 实时性能: {:.2f} frames/sec'.format(avg_time_per_frame_ms, pfs))
            f.write('\n')
            f.write('\n')
            f.close()
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('\n')
            f.write('\n')
            f.close()

        if self.args.camera:
            np.save(folder_path + 'metrics.npy', np.array([accuracy,precision,recall,f1,roc_auc]))
            # 将2D数组展平成1D数组，以便计算混淆矩阵
            pred_labels = preds
            true_labels = trues
            assert len(pred_labels) == len(true_labels), "True labels and predicted labels must have the same length"
            cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2, 3, 4, 5])
            # 创建混淆矩阵显示对象
            plot_multiclass_roc(trues, preds, 6, folder_path + 'roc_auc.png')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            # 绘制混淆矩阵
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(folder_path + 'confusion_matrix.png')
            # 显示混淆矩阵
            plt.show()

        else:
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def Tolabel(self, batch_y):
        # 去掉最后一个维度并转换为整数类型
        batch_y_squeezed = batch_y.long()
        # 确保传递给 F.one_hot 的张量是整数类型
        batch_y_one_hot = F.one_hot(batch_y_squeezed, num_classes=6)
        return batch_y_one_hot
    
def plot_multiclass_roc(trues, preds, num_classes, output_path='roc_auc.png'):        
    # 对真实标签进行one-hot编码
    y_test_bin = label_binarize(trues, classes=np.arange(num_classes))
    preds = label_binarize(preds, classes=np.arange(num_classes))
    # 为每个类别计算ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制所有类别的ROC曲线
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(num_classes), colors):
        RocCurveDisplay(fpr=fpr[i], tpr=tpr[i], roc_auc=roc_auc[i], estimator_name=f"Class {i}").plot(
            lw=2, color=color, ax=plt.gca())
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiclass Classification')
    
    # 保存图像
    plt.savefig(output_path)
    plt.close()
    
    print(f'ROC AUC 曲线已保存为: {output_path}')