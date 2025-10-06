import os
import torch
from models import Autoformer, Test_1_17_cross_1, Transformer, TimesNet, Nonstationary_Transformer, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, TimeXer, \
    DeepAR, WITRAN, LSTM, LSTMCNN, Official_V3, AdaWaveNet, \
    Official_F1, Official_F2, DeformTime, Test, Test_1, Test_2, Test_3, Test_1_1, Test_1_A1, Test_1_A2, Test_1_11, Test_1_2, Test_1_12, Test_1_13, \
    Test_1_14, Test_1_15, Test_1_16, Test_1_17, Test_1_17_cross_1, Test_1_17_cross_2, Test_1_17_damped_1, Test_1_19, Test_1_19_1, Test_1_19_2, Test_1_16_1, Test_1_18, \
    Test_1_17_cross_3, Test_1_17_cross_4, Test_1_16_1_embedding, Test_1_16_1_embedding_1, Test_1_16_1_embedding_1_1, Test_1_16_1_embedding_1_2, TCN, LightGBM, ARIMA, \
    Test_1_17_cross_4_ablation1, Test_1_17_cross_4_ablation2, Test_1_17_cross_4_ablation3, Test_1_17_cross_4_ablation4

from models.MLP import DLinear
import typing

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            "TimeXer": TimeXer,
            "DeepAR": DeepAR,
            "WITRAN": WITRAN,
            "LSTM": LSTM,
            "LSTMCNN": LSTMCNN,
            "Official_V3": Official_V3,
            "Official_F1": Official_F1,
            "Official_F2": Official_F2,
            "AdaWaveNet": AdaWaveNet,
            "DeformTime": DeformTime,
            "Test": Test,
            "Test_1": Test_1,
            "Test_2": Test_2,
            "Test_3": Test_3,
            "Test_1_1": Test_1_1,
            "Test_1_A1": Test_1_A1,
            "Test_1_A2": Test_1_A2,
            "Test_1_11": Test_1_11,
            "Test_1_2": Test_1_2,
            "Test_1_12": Test_1_12,
            "Test_1_13": Test_1_13,
            "Test_1_14": Test_1_14,
            "Test_1_15": Test_1_15,
            "Test_1_16": Test_1_16,
            "Test_1_17": Test_1_17,
            "Test_1_17_1": Test_1_17_cross_1,
            "Test_1_17_damped_1": Test_1_17_damped_1,
            "Test_1_19": Test_1_19,
            "Test_1_19_1": Test_1_19_1,
            "Test_1_19_2": Test_1_19_2,
            "Test_1_16_1": Test_1_16_1,
            "Test_1_18": Test_1_18,
            "Test_1_17_cross_2": Test_1_17_cross_2,
            "Test_1_17_cross_3": Test_1_17_cross_3,
            "Test_1_17_cross_4": Test_1_17_cross_4,
            "Test_1_16_1_embedding": Test_1_16_1_embedding,
            "Test_1_16_1_embedding_1": Test_1_16_1_embedding_1,
            "Test_1_16_1_embedding_1_1": Test_1_16_1_embedding_1_1,
            "Test_1_16_1_embedding_1_2": Test_1_16_1_embedding_1_2,
            "TCN": TCN,
            "LightGBM": LightGBM,
            "ARIMA": ARIMA,
            "Test_1_17_cross_4_ablation1": Test_1_17_cross_4_ablation1,
            "Test_1_17_cross_4_ablation2": Test_1_17_cross_4_ablation2,
            "Test_1_17_cross_4_ablation3": Test_1_17_cross_4_ablation3,
            "Test_1_17_cross_4_ablation4": Test_1_17_cross_4_ablation4
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
