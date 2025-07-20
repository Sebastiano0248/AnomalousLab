import subprocess

commands = [
    'python automated_trainer.py --trajectory_type "FractionalBrownianMotion" --parameter "hurst" --save_dir "best_models" --lr 0.00025 --batch_size 128 --epochs 30 --weight_decay 0.001 --num_particles 20000 --num_steps 200',
    'python automated_trainer.py --trajectory_type "ScaledBrownianMotion" --parameter "beta" --save_dir "best_models" --lr 0.0015 --batch_size 128 --epochs 20 --weight_decay 0.001 --num_particles 20000 --num_steps 500',
    'python automated_trainer.py --trajectory_type "AnnealedTransitTimeModel" --parameter "gamma_shape" --save_dir "best_models" --lr 0.00015 --batch_size 64 --epochs 20 --weight_decay 0.0001 --num_particles 15000 --num_steps 500',
    'python automated_trainer.py --trajectory_type "ContinuousTimeRandomWalk" --parameter "alpha" --save_dir "best_models" --lr 0.00025 --batch_size 128 --epochs 30 --weight_decay 0.001 --num_particles 20000 --num_steps 200',
    'python automated_trainer.py --trajectory_type "BrownianMotion" --parameter "sigma" --save_dir "best_models" --lr 0.00015 --batch_size 64 --epochs 20 --weight_decay 0.0001 --num_particles 15000 --num_steps 500',
    'python automated_trainer.py --trajectory_type "LevyWalks" --parameter "alpha" --save_dir "best_models" --lr 0.0005 --batch_size 32 --epochs 20 --weight_decay 0.001 --num_particles 10000 --num_steps 200',
    'python automated_trainer.py --trajectory_type "LevyWalks" --parameter "speed" --save_dir "best_models" --lr 0.0015 --batch_size 64 --epochs 20 --weight_decay 0.005 --num_particles 10000 --num_steps 300',
    'python automated_trainer.py --trajectory_type "BrownianMotion" --parameter "delta_t" --save_dir "best_models" --lr 0.0015 --batch_size 128 --epochs 20 --weight_decay 0.001 --num_particles 20000 --num_steps 500',
    'python automated_trainer.py --trajectory_type "LevyWalks" --parameter "beta" --save_dir "best_models" --lr 0.003 --batch_size 32 --epochs 15 --weight_decay 0.01 --num_particles 15000 --num_steps 200',
]

for cmd in commands:
    print(f"Running:\n{cmd}")
    subprocess.run(cmd, shell=True)
