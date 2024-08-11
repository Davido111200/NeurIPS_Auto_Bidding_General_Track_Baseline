import numpy as np
import random
from datetime import datetime
import torch
import wandb
import pandas as pd
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC
from bidding_train_env.strategy import BcBiddingStrategy
import logging
import ast
from run.run_evaluate import run_test

np.set_printoptions(suppress=True, precision=4)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

current_date = datetime.now().strftime("%Y-%m-%d")

def run_bc():
    """
    Run bc model training and evaluation.
    """
    train_model()
    # load_model()


def train_model():
    """
    train BC model
    """

    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)
    exp_prefix = f'BC-{current_date}-{random.randint(int(1e5), int(1e6) - 1)}'
    wandb.init(name=exp_prefix, group="BC", project="NeurIPS_Auto_Bidding")


    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)

    state_dim = 16
    normalize_indices = [13, 14, 15]
    is_normalize = True

    normalize_dic = normalize_state(training_data, state_dim, normalize_indices)
    normalize_reward(training_data, "reward_continuous")
    save_normalize_dict(normalize_dic, f"saved_model/{current_date}/BCtest")

    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    model = BC(dim_obs=state_dim)
    step_num = 100000
    batch_size = 100
    best_model_score = 0
    for i in range(step_num):
        states, actions, _, _, _ = replay_buffer.sample(batch_size)
        a_loss = model.step(states, actions)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")

        if i % 10000 == 0:
            agent = BcBiddingStrategy(load_model=False)
            agent.model = model
            agent.normalize_dict = normalize_dic
            all_reward, all_cost, cpa_real, cpa_constraint, score = run_test(agent)
            if score >= best_model_score:
                best_model_score = score
                best_model = model
            wandb.log({"training/action_loss": np.mean(a_loss)})
            wandb.log({"evaluate/all_reward": all_reward, "evaluate/all_cost": all_cost, "evaluate/cpa_real": cpa_real,
                       "evaluate/cpa_constraint": cpa_constraint, "evaluate/score": score})

    # model.save_net("saved_model/BCtest")
    best_model.save_jit(f"saved_model/{current_date}/BCtest")
    test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC(dim_obs=16)
    model.load_net("saved_model/BCtest")
    test_state = np.ones(16, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()
