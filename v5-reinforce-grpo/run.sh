python gen_data.py
python pretrain.py
python finetune.py

echo ""
echo "========== [1/3] REINFORCE (EMA) =========="
python reinforce_ema.py

echo ""
echo "========== [2/3] REINFORCE (RM) =========="
python train_reward_model.py
python reinforce_rm.py

echo ""
echo "========== [3/3] GRPO (group + KL) =========="
python reinforce_grpo.py --steps 1000

echo ""
echo ""
python compare.py
