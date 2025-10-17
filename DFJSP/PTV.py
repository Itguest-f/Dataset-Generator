import torch

def simulate_proc_time_variation(proc_times_batch, variation_rate, mode='gaussian'):
    """
    :param proc_times_batch: Tensor (batch, opes, mas)
    :param variation_rate: max ±20% disturbance
    :param variation_rate: max 20% disturbance
    :param mode: ['uniform', 'gaussian']
    :return: Tensor with varied processing times
    """
    if mode == 'uniform':
        noise = (torch.rand_like(proc_times_batch) - 0.5) * 2 * variation_rate
    elif mode == 'gaussian':
        noise = torch.randn_like(proc_times_batch) * variation_rate
    else:
        raise ValueError("Unsupported mode")
    noise = torch.clamp(noise, min = 0) # Speed improvement is in line with simulated scenarios
    # noise = torch.clamp(noise, min = -0.20) # Speed improvement is in line with simulated scenarios
    proc_times_varied = proc_times_batch * (1 + noise)
    return torch.clamp(proc_times_varied, min = 0)  # Avoid being negative


if __name__ == '__main__':
    example_proc_times = torch.tensor([
        [[10.0, 0],
         [8.0, 7.0],
         [6.0, 5.0]],

        [[15.0, 14.0],
         [9.0, 0],
         [10.0, 11.0]]
    ])
    nonzero_mask = example_proc_times != 0

    a = simulate_proc_time_variation(example_proc_times,variation_rate=0.5, mode='uniform')
    print(a)
