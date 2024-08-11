import numpy as np
import random
from datetime import datetime
import logging
import wandb
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.iql.iql import IQL
import sys
import pandas as pd
import ast
from run.run_evaluate import run_test
from bidding_train_env.strategy import IqlBiddingStrategy

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16

current_date = datetime.now().strftime("%Y-%m-%d")

def train_iql_model():
    """
    Train the IQL model.
    """
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    is_normalize = True

    if is_normalize:
        normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=[13, 14, 15])
        # select use continuous reward
        training_data['reward'] = normalize_reward(training_data, "reward_continuous")
        # select use sparse reward
        # training_data['reward'] = normalize_reward(training_data, "reward")
        save_normalize_dict(normalize_dic, f"saved_model/{current_date}/IQLtest")

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    # Train model
    model = IQL(dim_obs=STATE_DIM)
    # train_model_steps(model, replay_buffer)
    model = train_model_steps(model, replay_buffer, normalize_dic=normalize_dic)

    # Save model
    model.save_jit(f"saved_model/{current_date}/IQLtest")
    
    # Test trained model
    test_trained_model(model, replay_buffer)


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


def train_model_steps(model, replay_buffer, step_num=100000, batch_size=100, normalize_dic=None):
    best_model_score = 0
    best_model = None
    exp_prefix = f'IQL-{current_date}-{random.randint(int(1e5), int(1e6) - 1)}'
    wandb.init(name=exp_prefix, group="IQL", project="NeurIPS_Auto_Bidding")

    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)
        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)
        logger.info(f'Step: {i} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')

        if i % 10000 == 0:
            agent = IqlBiddingStrategy(load_model=False)
            agent.model = model
            agent.normalize_dict = normalize_dic
            all_reward, all_cost, cpa_real, cpa_constraint, score = run_test(agent)
            if score >= best_model_score:
                best_model_score = score
                best_model = model
            wandb.log({"training/Q_loss": q_loss, "training/V_loss": v_loss, "training/A_loss": a_loss})
            wandb.log({"evaluate/all_reward": all_reward, "evaluate/all_cost": all_cost, "evaluate/cpa_real": cpa_real,
                       "evaluate/cpa_constraint": cpa_constraint, "evaluate/score": score})
    return best_model


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred action:", tem)


def run_iql():
    print(sys.path)
    """
    Run IQL model training and evaluation.
    """
    train_iql_model()


if __name__ == '__main__':
    run_iql()
