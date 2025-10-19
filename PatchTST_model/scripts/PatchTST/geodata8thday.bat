@echo off

if not exist ".\logs" (
    mkdir .\logs
)

if not exist ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

set seq_len=7
set model_name=PatchTST
set root_path_name=.\
set data_path_name=geodata.csv
set model_id_name=geodata8thday
set data_name=custom
set random_seed=2021
set pred_len=1

python -u run_longExp.py ^
  --random_seed %random_seed% ^
  --is_training 1 ^
  --root_path %root_path_name% ^
  --data_path %data_path_name% ^
  --model_id %model_id_name%_%seq_len%_%pred_len% ^
  --model %model_name% ^
  --data %data_name% ^
  --features MS ^
  --seq_len %seq_len% ^
  --pred_len %pred_len% ^
  --enc_in 4 ^
  --e_layers 2 ^
  --n_heads 16 ^
  --d_model 128 ^
  --d_ff 256 ^
  --dropout 0.2 ^
  --fc_dropout 0.2 ^
  --head_dropout 0 ^
  --patch_len 2 ^
  --stride 1 ^
  --des Final ^
  --train_epochs 4 ^
  --patience 20 ^
  --itr 1 ^
  --batch_size 1 ^
  --learning_rate 0.0001 > logs\LongForecasting\%model_name%_%model_id_name%_%seq_len%_%pred_len%.log