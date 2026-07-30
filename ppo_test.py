if __name__ == "__main__":
    import time
    from ppo_parallel_model import train

    save = {}

    for mode in ["sync", "async"]:
        start = time.time()
        train(total_timesteps=300_000, minibatch_size=64, num_envs=64, vector_mode=mode,
              device_override='cpu', run_name=f"{mode}_test")
        save[f"{mode}_64envs"] = time.time() - start

    print(save)