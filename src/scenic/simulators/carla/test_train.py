from scenic.simulators.carla.train_utils import scenic_env

env = scenic_env("exp_uniform.scenic")

obs, info = env.reset()

for _ in range(100):
    action = [0, 0]
    obs, rew, te, tr, info = env.step(action)
    if te or tr:
        break

env.close()        
