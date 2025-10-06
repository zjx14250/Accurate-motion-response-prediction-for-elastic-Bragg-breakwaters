export CUDA_VISIBLE_DEVICES=0

model_name=Test_1_17_cross_4
seq_len=48
pred_len=48
model_id=exp_baseline_Mooring_tension

# 1. 训练

python -u run.py \
  --task_name forecasting_frequency_loss \
  --is_training 1 \
  --root_path ./dataset/wave/ \
  --data_path Hs=0.16_Tp=2.4.csv \
  --model_id $model_id \
  --model $model_name \
  --data WAVE \
  --features M \
  --target FlexGate2_Z \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 5 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 4 \
  --des 'Exp' \
  --itr 1 \
  --patience 5 \
  --auxi_loss "MAE" \
  --module_first 1 \
  --auxi_mode 'rfft' \
  --rec_lambda 0.6 \
  --auxi_lambda 0.4 \