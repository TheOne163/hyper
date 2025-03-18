import os
import subprocess
import sys

def main():
    # 环境配置
    env = "StarCraft2"
    map_name = "3m"
    algo = "rmappo"
    exp = "check"
    seed_max = 1

    print(f"env is {env}, map is {map_name}, algo is {algo}, exp is {exp}, max seed is {seed_max}")

    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 获取当前脚本所在目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建train_smac.py的完整路径
    train_script = os.path.join(current_dir, "..", "train", "train_smac.py")

    # 遍历种子值
    for seed in range(1, seed_max + 1):
        print(f"seed is {seed}:")
        
        # 构建命令行参数字符串
        cmd = f"{sys.executable} {train_script} " \
              f"--env_name {env} " \
              f"--algorithm_name {algo} " \
              f"--experiment_name {exp} " \
              f"--map_name {map_name} " \
              f"--seed {seed} " \
              f"--n_training_threads 1 " \
              f"--n_rollout_threads 1 " \
              f"--num_mini_batch 1 " \
              f"--episode_length 400 " \
              f"--num_env_steps 10000000 " \
              f"--ppo_epoch 15 " \
              f"--use_value_active_masks " \
              f"--use_eval " \
              f"--eval_episodes 32"

        try:
            # 执行命令
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 