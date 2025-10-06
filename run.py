import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_forecasting import Exp_Forecast
from exp.exp_forecasting_cycle import Exp_Forecast_Cycle
from exp.exp_forecasting_loss import Exp_Forecasting_Frequency_Loss
from exp.exp_forecasting_ml import Exp_Forecasting_ML
from exp.exp_forecasting_finetune import Exp_Forecasting_Finetune
from exp.exp_forecasting_loss_finetune import Exp_Forecasting_Loss_Finetune
from utils.print_args import print_args
import random
import numpy as np

if __name__ == "__main__":
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="TimesNet")

    # Personal Configuration
    parser.add_argument("--fanhua", action="store_true", help="inverse output data", default=False)  # 泛化测试 
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model') # 微调测试
    parser.add_argument("--manual_inverse", action="store_true", help="use manual inverse transform", default=False)  # 手动反归一化   
    parser.add_argument("--target_num", type=int, default=1, help="0：M-->M, 1：M-->N")  # 目标变量个数
    parser.add_argument("--f_dim", type=int, default=-4, help="feature dimension index for M->N prediction")  # 在 target_num=1 时，f_dim 为要预测的特征数
    parser.add_argument("--target", type=str, default="pitch", help="target feature in S or MS task") #  目标变量名称/N个目标变量名称的最后一个

    # basic config
    parser.add_argument("--task_name", type=str, required=True, default="long_term_forecast", help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]")
    parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
    parser.add_argument("--predict_attn", action="store_true", help="run prediction to get attention weights and save as .npy", default=False)
    parser.add_argument("--model_id", type=str, required=True, default="test", help="model id")
    parser.add_argument("--model", type=str, required=True, default="Autoformer", help="model name, options: [Autoformer, Transformer, TimesNet]")

    # data loader
    parser.add_argument("--data", type=str, required=True, default="ETTm1", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./data/ETT/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument("--features",type=str,default="M",help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate")
    parser.add_argument("--freq",type=str,default="h",help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h")
    parser.add_argument("--checkpoints",type=str,default="./checkpoints/",help="location of model checkpoints")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length"
)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="subset for M4")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument("--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)")

    # model define
    parser.add_argument("--expand", type=int, default=2, help="expansion factor for Mamba")
    parser.add_argument("--d_conv", type=int, default=4, help="conv kernel size for Mamba")
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--last_out", type=int, default=7, help="last output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--distil",action="store_false",help="whether to use distilling in encoder, using this argument means not using distilling",default=True,)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--channel_independence", type=int, default=1, help="0: channel dependence 1: channel independence for FreTS model")
    parser.add_argument("--decomp_method", type=str, default="moving_avg", help="method of series decompsition, only support moving_avg or dft_decomp")
    parser.add_argument("--use_norm", type=int, default=1, help="whether to use normalize; True 1 False 0")
    parser.add_argument("--down_sampling_layers", type=int, default=0, help="num of down sampling layers")
    parser.add_argument("--down_sampling_window", type=int, default=1, help="down sampling window size")
    parser.add_argument("--down_sampling_method", type=str, default=None, help="down sampling method, only support avg, max, conv")
    parser.add_argument("--seg_len", type=int, default=48, help="the length of segmen-wise iteration of SegRNN")

    # Add different model's own parameters
    # Test_1_14
    
    # AdaWaveNet
    parser.add_argument("--lifting_levels", type=int, default=3, help="num of lifting levels")
    parser.add_argument("--lifting_kernel_size", type=int, default=7, help="lifting kernel size")
    parser.add_argument("--n_clusters", type=int, default=4, help="num of clusters")
    parser.add_argument('--regu_details', type=float, default=0.01, help='regu_details of lifting scheme')
    parser.add_argument('--regu_approx', type=float, default=0.01, help='regu_approx of lifting scheme')
    
    # Official
    parser.add_argument("--self_n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--cross_n_heads", type=int, default=8, help="num of heads")

    # ForFEDTest411_X1
    parser.add_argument("--mask_ratio", type=float, default=1.0, help="mask ratio")

    # For CycleNet.
    parser.add_argument('--cycle', type=int, default=24, help='cycle length')

    # For DeepAR
    parser.add_argument('--hidden_dim', type=int, default=40, help='hidden dimension')
    
    # For WITRAN
    parser.add_argument('--WITRAN_deal', type=str, default='None', 
        help='WITRAN deal data type, options:[None, standard]')
    parser.add_argument('--WITRAN_grid_cols', type=int, default=24, 
        help='Numbers of data grid cols for WITRAN')
    
    # FreDF
    parser.add_argument('--rec_lambda', type=float, default=0.5, help='weight of reconstruction function')         # 时域权重
    parser.add_argument('--auxi_lambda', type=float, default=0.5, help='weight of auxilary function')               # 频域权重
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft, welch]')
    parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')
    # welch params
    parser.add_argument('--welch_nperseg', type=int, default=64, help='segment length for Welch PSD')
    parser.add_argument('--welch_noverlap', type=int, default=32, help='overlap length for Welch PSD')
    # fre-loss
    parser.add_argument('--add_noise', action='store_true', help='add noise')
    parser.add_argument('--noise_amp', type=float, default=1, help='noise ampitude')
    parser.add_argument('--noise_freq_percentage', type=float, default=0.05, help='noise frequency percentage')
    parser.add_argument('--noise_seed', type=int, default=2023, help='noise seed')
    parser.add_argument('--noise_type', type=str, default='sin', help='noise type, options: [sin, normal]')
    parser.add_argument('--cutoff_freq_percentage', type=float, default=0.06, help='cutoff frequency')
    
    # DeformTime parameters
    parser.add_argument('--kernel', type=int, default=6, help='kernel size')
    parser.add_argument('--n_reshape', type=int, default=16)
    parser.add_argument('--layer_dropout', type=float, default=0.6, help='path   dropout')
    parser.add_argument('--stride', type=int, default=4, help='stride when splitting')
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')

    # optimization
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=50, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp",action="store_true",help="use automatic mixed precision training",default=False)
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

    # de-stationary projector params
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128], help="hidden layer dimensions of projector (List)")
    parser.add_argument("--p_hidden_layers", type=int, default=2, help="number of hidden layers in projector")

    # metrics (dtw)
    parser.add_argument("--use_dtw", type=bool, default=False, help="the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)")

    # Augmentation
    parser.add_argument("--augmentation_ratio", type=int, default=0, help="How many times to augment")
    parser.add_argument("--seed", type=int, default=2, help="Randomization seed")
    parser.add_argument("--jitter", default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument("--scaling", default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument("--permutation", default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument("--randompermutation", default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument("--magwarp", default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument("--timewarp", default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument("--windowslice", default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument("--windowwarp", default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument("--rotation", default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument("--spawner", default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument("--dtwwarp", default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument("--shapedtwwarp", default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument("--wdba", default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument("--discdtw", default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument("--discsdtw", default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument("--extra_tag", type=str, default="", help="Anything extra")

    # --- Parameters for the GAT-based WavePlateModel (formerly Test.py) ---
    parser.add_argument('--H_t', type=int, default=32, help='Hidden dimension for TimeBackbone CNN')
    parser.add_argument('--F_f', type=int, default=16, help='Hidden dimension for FilterBank CNN')
    parser.add_argument('--filter_k', type=int, default=9, help='Kernel size for FilterBank CNN')
    parser.add_argument('--plate_nodes', type=int, default=2, help='Number of plate nodes in the graph')
    parser.add_argument('--gat_out', type=int, default=64, help='Output dimension for GAT layer')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads for GAT layer')

    # New parameters
    parser.add_argument('--loss_weight_x', type=float, default=0.5, help='weight for X-direction loss component')
    parser.add_argument('--loss_weight_z', type=float, default=0.5, help='weight for Z-direction loss component')

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    exp = None  # 初始化exp变量
    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    elif args.task_name == "forecasting":
        Exp = Exp_Forecast
    elif args.task_name == "forecasting_cycle":
        Exp = Exp_Forecast_Cycle
    elif args.task_name == "forecasting_frequency_loss":
        Exp = Exp_Forecasting_Frequency_Loss
    elif args.task_name == "forecasting_ml":
        Exp = Exp_Forecasting_ML
    elif args.task_name == "forecasting_finetune":
        Exp = Exp_Forecasting_Finetune
    elif args.task_name == "forecasting_loss_finetune":
        Exp = Exp_Forecasting_Loss_Finetune
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
            )

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        exp = Exp(args)  # set experiments
        if args.predict_attn:
            print(">>>>>>>predicting attn weights: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.predict(setting, load=True)
        else:
            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
