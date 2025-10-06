from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric_wave_spectral
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import logging
from dtaidistance import dtw_ndim
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)

warnings.filterwarnings("ignore")


class Exp_Forecasting_Frequency_Loss(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecasting_Frequency_Loss, self).__init__(args)
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件输出将在train函数中设置
        self.file_handler = None

        # 添加 mask 初始化逻辑
        self.pred_len = args.pred_len

        if args.add_noise and args.noise_amp > 0:
            seq_len = args.pred_len
            cutoff_freq_percentage = args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None

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
        criterion = nn.MSELoss(reduction='mean')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.features == "MS":
                    f_dim = -1
                # 单变量预测单变量
                elif self.args.features == "S":
                    f_dim = 0
                # 多变量预测单变量
                elif self.args.features == "M" and self.args.target_num == 1:
                    f_dim = self.args.f_dim
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        # 保存训练数据的 scaler 参数
        scale_params = {
            'mean_': train_data.scaler.mean_,
            'scale_': train_data.scaler.scale_
        }
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 设置日志文件路径
        if self.file_handler is None:
            log_path = os.path.join(path, 'training.log')
            self.file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)

        # 记录实验参数到日志,使用格式化输出
        self.logger.info("\nArgs in experiment:")
        self.logger.info("Basic Config")
        self.logger.info(f"  Task Name:          {self.args.task_name:<18} Is Training:        {self.args.is_training:<18}")
        self.logger.info(f"  Model ID:           {self.args.model_id:<18} Model:              {self.args.model:<18}")
        
        self.logger.info("\nData Loader")
        self.logger.info(f"  Data:               {self.args.data:<18} Root Path:          {self.args.root_path:<18}")
        self.logger.info(f"  Data Path:          {self.args.data_path:<18} Features:           {self.args.features:<18}")
        self.logger.info(f"  Target:             {self.args.target:<18} Freq:               {self.args.freq:<18}")
        self.logger.info(f"  Checkpoints:        {self.args.checkpoints:<18}")
        
        self.logger.info("\nModel Parameters")
        self.logger.info(f"  Top k:              {self.args.top_k:<18} Num Kernels:        {self.args.num_kernels:<18}")
        self.logger.info(f"  Enc In:             {self.args.enc_in:<18} Dec In:             {self.args.dec_in:<18}")
        self.logger.info(f"  C Out:              {self.args.c_out:<18} d model:            {self.args.d_model:<18}")
        self.logger.info(f"  n heads:            {self.args.n_heads:<18} e layers:           {self.args.e_layers:<18}")
        self.logger.info(f"  d layers:           {self.args.d_layers:<18} d FF:               {self.args.d_ff:<18}")
        self.logger.info(f"  Moving Avg:         {self.args.moving_avg:<18} Factor:             {self.args.factor:<18}")
        self.logger.info(f"  Distil:             {int(self.args.distil):<18} Dropout:            {self.args.dropout:<18}")
        self.logger.info(f"  Embed:              {self.args.embed:<18} Activation:         {self.args.activation:<18}")

        self.logger.info("\nRun Parameters")
        self.logger.info(f"  Num Workers:        {self.args.num_workers:<18} Itr:                {self.args.itr:<18}")
        self.logger.info(f"  Train Epochs:       {self.args.train_epochs:<18} Batch Size:         {self.args.batch_size:<18}")
        self.logger.info(f"  Patience:           {self.args.patience:<18} Learning Rate:      {self.args.learning_rate:<18}")
        self.logger.info(f"  Des:                {self.args.des:<18} Loss:               {self.args.loss:<18}")
        self.logger.info(f"  Lradj:              {self.args.lradj:<18} Use Amp:            {int(self.args.use_amp):<18}")

        self.logger.info("\nGPU")
        self.logger.info(f"  Use GPU:            {int(self.args.use_gpu):<18} GPU:                {self.args.gpu:<18}")
        self.logger.info(f"  Use Multi GPU:      {int(self.args.use_multi_gpu):<18} Devices:            {self.args.devices:<18}")

        self.logger.info("\nDe-stationary Projector Params")
        self.logger.info(f"  P Hidden Dims:      {', '.join(map(str, self.args.p_hidden_dims)):<18} P Hidden Layers:    {self.args.p_hidden_layers}")
        self.logger.info("\n" + "-" * 80 + "\n")  # 添加分隔线

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.logger.info(f"Start training for setting: {setting}")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        if self.args.features == "MS": f_dim = -1
                        elif self.args.features == "S": f_dim = 0
                        elif self.args.features == "M" and self.args.target_num == 1: f_dim = self.args.f_dim
                        else: f_dim = 0

                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                        
                        loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item())

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if self.args.features == "MS": f_dim = -1
                    elif self.args.features == "S": f_dim = 0
                    elif self.args.features == "M" and self.args.target_num == 1: f_dim = self.args.f_dim
                    else: f_dim = 0

                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    loss = 0
                    if self.args.rec_lambda:
                        loss_rec = criterion(outputs, batch_y)
                        
                        loss += self.args.rec_lambda * loss_rec
                        if (i + 1) % 100 == 0:
                            print(f"\tloss_rec: {loss_rec.item()}")

                        # self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss_rec, self.step)

                    if self.args.auxi_lambda:
                        # fft shape: [B, P, D]
                        if self.args.auxi_mode == "fft":
                            loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)

                        elif self.args.auxi_mode == "rfft":
                            if self.args.auxi_type == 'complex':
                                loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                            elif self.args.auxi_type == 'complex-phase':
                                loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                            elif self.args.auxi_type == 'complex-mag-phase':
                                loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                                loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                            elif self.args.auxi_type == 'phase':
                                loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                            elif self.args.auxi_type == 'mag':
                                loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                            elif self.args.auxi_type == 'mag-phase':
                                loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                                loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                            else:
                                raise NotImplementedError

                        elif self.args.auxi_mode == "rfft-D":
                            loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)

                        elif self.args.auxi_mode == "rfft-2D":
                            loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)
                        
                        elif self.args.auxi_mode == "legendre":
                            loss_auxi = leg_torch(outputs, self.args.leg_degree, device=self.device) - leg_torch(batch_y, self.args.leg_degree, device=self.device)
                        
                        elif self.args.auxi_mode == "chebyshev":
                            loss_auxi = chebyshev_torch(outputs, self.args.leg_degree, device=self.device) - chebyshev_torch(batch_y, self.args.leg_degree, device=self.device)
                        
                        elif self.args.auxi_mode == "hermite":
                            loss_auxi = hermite_torch(outputs, self.args.leg_degree, device=self.device) - hermite_torch(batch_y, self.args.leg_degree, device=self.device)
                        
                        elif self.args.auxi_mode == "laguerre":
                            loss_auxi = laguerre_torch(outputs, self.args.leg_degree, device=self.device) - laguerre_torch(batch_y, self.args.leg_degree, device=self.device)
                        else:
                            raise NotImplementedError

                        if self.mask is not None:
                            loss_auxi *= self.mask

                        if self.args.auxi_loss == "MAE":
                            # MAE, 最小化element-wise error的模长
                            loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                        elif self.args.auxi_loss == "MSE":
                            # MSE, 最小化element-wise error的模长
                            loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                        else:
                            raise NotImplementedError

                        loss += self.args.auxi_lambda * loss_auxi
                        if (i + 1) % 100 == 0:
                            print(f"\tloss_auxi: {loss_auxi.item()}")

                        # self.logger.info(f'{self.pred_len}/train/loss_auxi', loss_auxi, epoch)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.logger.info(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    self.logger.info(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            self.logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        np.savez(path + "/" + "scaler_params.npz", **scale_params)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        # 获取data_path作为结果文件标识
        data_identifier = os.path.splitext(os.path.basename(self.args.data_path))[0]
        result_folder = os.path.join(self.args.checkpoints, setting, data_identifier)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.features == "MS":
                    f_dim = -1
                # 单变量预测单变量
                elif self.args.features == "S":
                    f_dim = 0
                # 多变量预测单变量
                elif self.args.features == "M" and self.args.target_num == 1:
                    f_dim = self.args.f_dim
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    if not self.args.manual_inverse:                     # 使用自带的inverse_transform方法
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    else:                                                # 使用手动反归一化（针对外生变量方法）
                        output_scale = test_data.scaler.scale_[-4:]
                        output_mean = test_data.scaler.mean_[-4:]
                        batch_y_scale = test_data.scaler.scale_
                        batch_y_mean = test_data.scaler.mean_
                        
                        if outputs.shape[-1] != len(output_scale):
                            raise ValueError(f"输出特征数量 {outputs.shape[-1]} 与 scaler 拟合特征数量 {len(output_scale)} 不一致。")
                        
                        outputs = outputs * output_scale + output_mean
                        batch_y = batch_y * batch_y_scale + batch_y_mean

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # 分别提取X和Z方向的数据
        x_indices = [0, 2]  # flexgate1_X和flexgate2_X的索引
        z_indices = [1, 3]  # flexgate1_Z和flexgate2_Z的索引
        
        preds_x = preds[:, :, x_indices]
        trues_x = trues[:, :, x_indices]
        preds_z = preds[:, :, z_indices]
        trues_z = trues[:, :, z_indices]

        # dtw calculation - 总体
        if self.args.use_dtw:
            dtw_list = []
            
            for i in range(preds.shape[0]):
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                
                # 提取当前样本的所有维度
                pred_sample = preds[i]  # 形状: [时间步长, 变量数]
                true_sample = trues[i]  # 形状: [时间步长, 变量数]
                
                # 计算多维DTW
                distance = dtw_ndim.distance(pred_sample, true_sample)
                dtw_list.append(distance)
            
            dtw = np.array(dtw_list).mean()
        else:
            dtw = "not calculated"

        # dtw calculation - X方向
        if self.args.use_dtw:
            dtw_x_list = []
            
            for i in range(preds_x.shape[0]):
                if i % 100 == 0:
                    print("calculating dtw X iter:", i)
                
                pred_sample_x = preds_x[i]  # 形状: [时间步长, X变量数]
                true_sample_x = trues_x[i]
                
                distance_x = dtw_ndim.distance(pred_sample_x, true_sample_x)
                dtw_x_list.append(distance_x)
            
            dtw_x = np.array(dtw_x_list).mean()
        else:
            dtw_x = "not calculated"

        # dtw calculation - Z方向
        if self.args.use_dtw:
            dtw_z_list = []
            
            for i in range(preds_z.shape[0]):
                if i % 100 == 0:
                    print("calculating dtw Z iter:", i)
                
                pred_sample_z = preds_z[i]  # 形状: [时间步长, Z变量数]
                true_sample_z = trues_z[i]
                
                distance_z = dtw_ndim.distance(pred_sample_z, true_sample_z)
                dtw_z_list.append(distance_z)
            
            dtw_z = np.array(dtw_z_list).mean()
        else:
            dtw_z = "not calculated"

        # 计算总体指标（使用新的频谱指标）
        mae, rmse, rr, spectral_mae = metric_wave_spectral(preds, trues)
        
        # 计算X方向指标
        mae_x, rmse_x, rr_x, spectral_mae_x = metric_wave_spectral(preds_x, trues_x)
        
        # 计算Z方向指标
        mae_z, rmse_z, rr_z, spectral_mae_z = metric_wave_spectral(preds_z, trues_z)

        # 保存结果到文件
        result_file = os.path.join(result_folder, "result.txt")
        f = open(result_file, "a")
        f.write(setting + "  \n")
        f.write("Overall - mae:{}, rmse:{}, rr:{}, spectral_mae:{}, dtw:{}".format(
            mae, rmse, rr, spectral_mae, dtw))
        f.write("\n")
        f.write("X Direction - mae:{}, rmse:{}, rr:{}, spectral_mae:{}, dtw:{}".format(
            mae_x, rmse_x, rr_x, spectral_mae_x, dtw_x))
        f.write("\n")
        f.write("Z Direction - mae:{}, rmse:{}, rr:{}, spectral_mae:{}, dtw:{}".format(
            mae_z, rmse_z, rr_z, spectral_mae_z, dtw_z))
        f.write("\n")
        f.write("\n")
        f.close()

        # 保存numpy数组
        np.save(os.path.join(result_folder, "metrics.npy"), np.array([mae, rmse, rr, spectral_mae]))
        np.save(os.path.join(result_folder, "metrics_x.npy"), np.array([mae_x, rmse_x, rr_x, spectral_mae_x]))
        np.save(os.path.join(result_folder, "metrics_z.npy"), np.array([mae_z, rmse_z, rr_z, spectral_mae_z]))
        np.save(os.path.join(result_folder, "pred.npy"), preds)
        np.save(os.path.join(result_folder, "true.npy"), trues)
        np.save(os.path.join(result_folder, "pred_x.npy"), preds_x)
        np.save(os.path.join(result_folder, "true_x.npy"), trues_x)
        np.save(os.path.join(result_folder, "pred_z.npy"), preds_z)
        np.save(os.path.join(result_folder, "true_z.npy"), trues_z)

        # 日志输出
        self.logger.info(f"Start testing for setting: {setting}")
        self.logger.info("test shape: {}, {}".format(preds.shape, trues.shape))
        
        self.logger.info('Overall - mae:{:.6f}, rmse:{:.6f}, rr:{:.6f}, spectral_mae:{:.6f}, dtw:{}'.format(
            mae, rmse, rr, spectral_mae, dtw))
        self.logger.info('X Direction - mae:{:.6f}, rmse:{:.6f}, rr:{:.6f}, spectral_mae:{:.6f}, dtw:{}'.format(
            mae_x, rmse_x, rr_x, spectral_mae_x, dtw_x))
        self.logger.info('Z Direction - mae:{:.6f}, rmse:{:.6f}, rr:{:.6f}, spectral_mae:{:.6f}, dtw:{}'.format(
            mae_z, rmse_z, rr_z, spectral_mae_z, dtw_z))
        return

    def predict(self, setting, load=True):
        test_data, test_loader = self._get_data(flag="test")
        if load:
            self.logger.info("loading model for prediction")
            model_path = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
            if not os.path.exists(model_path):
                self.logger.error(f"Checkpoint not found at {model_path}")
                raise FileNotFoundError(f"Checkpoint not found at {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # 获取data_path作为结果文件标识
        data_identifier = os.path.splitext(os.path.basename(self.args.data_path))[0]
        result_folder = os.path.join(self.args.checkpoints, setting, data_identifier)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        all_attn_weights = []
        self.model.eval()
        
        self.logger.info(f"Starting prediction to get attention weights for {setting}")

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processing batch {i+1}/{len(test_loader)}...")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Get attention weights
                model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                try:
                    # 获取自注意力权重
                    self_attn_weights = model_to_use.decoder.layers[1].attn
                    
                    # 获取交叉注意力权重
                    cross_attn_fwd_weights = model_to_use.decoder.layers[1].cross_attn_fwd
                    cross_attn_rev_weights = model_to_use.decoder.layers[1].cross_attn_rev
                    
                    # 保存所有三种注意力权重到字典
                    all_attn_weights.append({
                        'self_attn': self_attn_weights.detach().cpu().numpy(),
                        'cross_attn_fwd': cross_attn_fwd_weights.detach().cpu().numpy(), 
                        'cross_attn_rev': cross_attn_rev_weights.detach().cpu().numpy()
                    })
                    
                except AttributeError as e:
                    self.logger.error(f"Could not find attention weights: {e}")
                    self.logger.error("Please ensure you have modified the model to save cross-attention weights.")
                    return

        # Concatenate and save
        if not all_attn_weights:
            self.logger.warning("No attention weights were collected. Nothing to save.")
            return

        # 分别提取和保存不同类型的注意力权重
        self_attn_all = [item['self_attn'] for item in all_attn_weights]
        cross_fwd_attn_all = [item['cross_attn_fwd'] for item in all_attn_weights]
        cross_rev_attn_all = [item['cross_attn_rev'] for item in all_attn_weights]
        
        # 拼接各类型的注意力权重
        self_attn_weights_concat = np.concatenate(self_attn_all, axis=0)
        cross_fwd_attn_weights_concat = np.concatenate(cross_fwd_attn_all, axis=0)
        cross_rev_attn_weights_concat = np.concatenate(cross_rev_attn_all, axis=0)
        
        # 保存文件
        np.save(os.path.join(result_folder, "self_attn_weights.npy"), self_attn_weights_concat)
        np.save(os.path.join(result_folder, "cross_attn_fwd_weights.npy"), cross_fwd_attn_weights_concat)
        np.save(os.path.join(result_folder, "cross_attn_rev_weights.npy"), cross_rev_attn_weights_concat)
        
        # 为了兼容原有的可视化代码，也保存自注意力权重为默认的 attn_weights.npy
        np.save(os.path.join(result_folder, "attn_weights.npy"), self_attn_weights_concat)

        self.logger.info(f"Prediction complete. All attention weights saved.")
        self.logger.info(f"  - Self attention shape: {self_attn_weights_concat.shape}")
        self.logger.info(f"  - Cross attention forward shape: {cross_fwd_attn_weights_concat.shape}")
        self.logger.info(f"  - Cross attention reverse shape: {cross_rev_attn_weights_concat.shape}")
        self.logger.info(f"  - Files saved in: {result_folder}")
        
        return
    



