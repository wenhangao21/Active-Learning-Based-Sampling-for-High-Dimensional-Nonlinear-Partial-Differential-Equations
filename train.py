import torch
from torch import optim, autograd
import timeit
from numpy import zeros
from utilities import (
    generate_points_in_the_domain, generate_points_on_the_boundary,
    gradients, true_solution, get_l2_error, get_testing_points
)
from sampling import importance_on_boundary, importance_in_domain
import network_setting
import scipy.io

# Set to run on CUDA
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

# Hyperparameters
n_epoch = 50000
trials = 30
end_time = 300
N_1, N_2 = 1024, 1024
m_drm_dgm = 20
lr = 3e-3
lambda_term = 5e6

# Pre-allocate arrays to store results
l2error_different_seeds = zeros((trials,))
time_different_seeds = zeros((trials,))
x_test = get_testing_points()
true_solution_test = true_solution(x_test)


def main():
    error_dict, time_dict = [], []

    for k in range(trials):
        print(f'Trial: {k + 1}')
        torch.manual_seed(k + 1)

        model = network_setting.drnn(m_drm_dgm).cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)

        time_elapsed = 0
        timeseq = zeros((n_epoch,))
        l2seq = zeros((n_epoch,))

        for epoch in range(n_epoch):
            if time_elapsed > end_time:
                l2error_different_seeds[k] = l2_error
                time_different_seeds[k] = time_elapsed
                break

            start_time = timeit.default_timer()

            x_1 = generate_points_in_the_domain(N_1, 2).cuda().requires_grad_()
            x_2 = importance_on_boundary(N_2, model, 1).cuda()
            network_solution_1 = model(x_1)
            network_solution_2 = model(x_2)
            true_solution_2 = true_solution(x_2)

            # Calculate gradients and losses
            nabla = autograd.grad(
                outputs=network_solution_1, inputs=x_1,
                grad_outputs=torch.ones_like(network_solution_1),
                create_graph=True, only_inputs=True
            )[0]
            delta = gradients(x_1, nabla[:, :1])[:, :1] + gradients(x_1, nabla[:, 1:2])[:, 1:2]
            loss_1 = torch.mean((delta - 2) ** 2)
            loss_2 = lambda_term * torch.mean((network_solution_2 - true_solution_2) ** 2)
            loss = loss_1 + loss_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            time_elapsed += timeit.default_timer() - start_time
            timeseq[epoch] = time_elapsed

            l2_error = get_l2_error(model(x_test.cuda()), true_solution_test.cuda()).item()
            l2seq[epoch] = l2_error

            if epoch % 500 == 0:
                print(f'Time: {time_elapsed:.2f}, Epoch: {epoch}, Loss: {loss.item():.4f}, Error: {l2_error:.4f}')

        error_dict.append(l2seq)
        time_dict.append(timeseq)

    # Calculate averages and save to .mat
    l2_average_seq = sum(error_dict) / trials
    time_average_seq = sum(time_dict) / trials
    scipy.io.savemat('er_average_error.mat', {'er_average_error': l2_average_seq})
    scipy.io.savemat('er_average_time.mat', {'er_average_time': time_average_seq})
    scipy.io.savemat('er_l2error_different_seeds.mat', {'er_l2error_different_seeds': l2error_different_seeds})
    scipy.io.savemat('er_time_different_seeds.mat', {'er_time_different_seeds': time_different_seeds})

    # Save mean and std to txt
    mean_error = l2error_different_seeds.mean()
    std_error = l2error_different_seeds.std()
    mean_time = time_different_seeds.mean()
    std_time = time_different_seeds.std()

    with open('results_summary.txt', 'w') as f:
        f.write(f"L2 Error - Mean: {mean_error}, Std: {std_error}\n")
        f.write(f"Time - Mean: {mean_time}, Std: {std_time}\n")


if __name__ == '__main__':
    main()


