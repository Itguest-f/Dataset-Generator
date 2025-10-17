import random

import torch


def new_jobs_arrive_time(batch_size, num_jobs, num_new_jobs, eva):
    """
    高效生成包含旧工件占位和新工件指数分布到达时间的张量。

    参数:
    - batch_size: 批大小
    - num_jobs: 当前旧工件数量
    - num_new_jobs: 新增工件数量
    - eva: 到达时间的指数分布均值 (Eva)

    返回:
    - Tensor: [batch_size, num_jobs + num_new_jobs] 的到达时间张量
    """
    A0 = torch.zeros(size=(batch_size, num_jobs), dtype=torch.long)  # 旧工件占位
    A1 = torch.distributions.Exponential(rate=1 / eva).sample((num_new_jobs,))  # 新工件到达时间
    A1, _ = torch.sort(A1)  # 排序（可选）
    A1 = A1.expand(batch_size, -1)  # 扩展到每个 batch
    return torch.cat((A0, A1), dim=1)  # 合并

def generate_new_jobs(num_jobs, num_mas,proctime_min=1, proctime_max=20, dev_ratio=0.2):
    """
    生成新工件的 .fjs 格式行列表，每行对应一个工件
    返回：List[str]
    """
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
    nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
    num_opes = sum(nums_ope)

    nums_option = [random.randint(1, num_mas) for _ in range(num_opes)]
    ope_ma = []
    for val in nums_option:
        ope_ma.extend(sorted(random.sample(range(1, num_mas + 1), val)))

    proc_times_mean = [random.randint(proctime_min, proctime_max) for _ in range(num_opes)]
    proc_time = []
    for i in range(len(nums_option)):
        low = max(proctime_min, round(proc_times_mean[i] * (1 - dev_ratio)))
        high = min(proctime_max, round(proc_times_mean[i] * (1 + dev_ratio)))
        proc_time.extend([random.randint(low, high) for _ in range(nums_option[i])])

    num_ope_bias = [sum(nums_ope[0:i]) for i in range(num_jobs)]
    num_ma_bias = [sum(nums_option[0:i]) for i in range(num_opes)]

    lines = []
    for i in range(num_jobs):
        flag = 0
        flag_time = 0
        flag_new_ope = 1
        idx_ope = -1
        idx_ma = 0
        line = []
        option_max = sum(nums_option[num_ope_bias[i]:(num_ope_bias[i] + nums_ope[i])])
        idx_option = 0
        while True:
            if flag == 0:
                line.append(nums_ope[i])
                flag += 1
            elif flag == flag_new_ope:
                idx_ope += 1
                idx_ma = 0
                flag_new_ope += nums_option[num_ope_bias[i] + idx_ope] * 2 + 1
                line.append(nums_option[num_ope_bias[i] + idx_ope])
                flag += 1
            elif flag_time == 0:
                line.append(ope_ma[num_ma_bias[num_ope_bias[i] + idx_ope] + idx_ma])
                flag += 1
                flag_time = 1
            else:
                line.append(proc_time[num_ma_bias[num_ope_bias[i] + idx_ope] + idx_ma])
                flag += 1
                flag_time = 0
                idx_option += 1
                idx_ma += 1
            if idx_option == option_max:
                str_line = " ".join([str(val) for val in line])
                lines.append(str_line + '\n')
                break
    return lines


if __name__ == '__main__':
    lines = generate_new_jobs(2,5,1,19)
    print(lines)
    A = new_jobs_arrive_time(6,10,3,50)
    print(A)