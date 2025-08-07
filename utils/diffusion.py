import torch

def pyramid_noise_like(trajectory, discount=0.9):
    # refer to https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31
    b, n, c = trajectory.shape # EDIT: w and h get over-written, rename for a different variant!
    trajectory_reshape = trajectory.permute(0, 2, 1)
    up_sample = torch.nn.Upsample(size=(n), mode='linear')
    noise = torch.randn_like(trajectory_reshape)
    for i in range(10):
        r = torch.rand(1, device=trajectory.device) + 1  # Rather than always going 2x,
        n = max(1, int(n/(r**i)))
        # print(i, n)
        noise += up_sample(torch.randn(b, c, n).to(trajectory_reshape)) * discount**i
        if n==1: break # Lowest resolution is 1x1
    # print(noise, noise/noise.std())
    noise = noise.permute(0, 2, 1)
    return (noise/noise.std()).float()

def get_s_from_o(edge_index, o_index):
    # get object from subject
    target_value = o_index  
    mask = edge_index[1, :] == target_value  
    return edge_index[0, mask] 