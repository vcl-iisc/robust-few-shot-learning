## CIFAR-FS 1-SHOT
python code/finetune.py --dataset CIFAR-FS --architecture ResNet12 --way 5 --support_shot 1 --query_shot 15 --num_episodes 1000 --attack 'pgd' --reg 'cosine_lp_q' --r_max 16 --mode 'curr' --run_name 'cosine_lp_teacher_init_curr16-2' --gpu 2

## CIFAR-FS 5-SHOT
python code/finetune.py --dataset CIFAR-FS --architecture ResNet12 --way 5 --support_shot 5 --query_shot 15 --num_episodes 1000 --attack 'pgd' --reg 'cosine_lp_q' --r_max 16 --mode 'curr' --run_name 'cosine_lp_teacher_init_curr16-2' --gpu 2