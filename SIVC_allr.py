import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def read_FSIVCqt_auto(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 自动推断 size, ntau, n_dot
    size = 0
    ntau = -1
    n_dot = 0
    idx = 0

    # 统计 size 和 ntau
    while idx < len(lines):
        if lines[idx].startswith("rank"):
            size += 1
            idx += 1
            # 每个 rank 下，统计 itau 和 n_dot
            itau_count = 0
            while idx < len(lines) and lines[idx].startswith("itau"):
                itau_count += 1
                idx += 1
                # 跳过 n_dot 数值
                n_dot_count = 0
                while idx < len(lines) and not lines[idx].startswith("rank") and not lines[idx].startswith("itau"):
                    n_dot_count += 1
                    idx += 1
                if n_dot_count > n_dot:
                    n_dot = n_dot_count
            if ntau == -1:
                ntau = itau_count - 1  # -1 因为 itau 从 0 开始
        else:
            idx += 1  # 跳过非 rank 行
    
    # 创建存储数组 SIVC[rank, itau, iq]
    SIVC = np.zeros((size, ntau + 1, n_dot), dtype=float)

    # 重新读取文件并填充数据
    idx = 0
    rank_idx = -1
    while idx < len(lines):
        if lines[idx].startswith("rank"):
            rank_idx += 1
            idx += 1
            itau_idx = -1
            while idx < len(lines) and lines[idx].startswith("itau"):
                itau_idx += 1
                idx += 1
                iq_idx = 0
                while idx < len(lines) and not lines[idx].startswith("rank") and not lines[idx].startswith("itau"):
                    SIVC[rank_idx, itau_idx, iq_idx] = float(lines[idx].strip())
                    iq_idx += 1
                    idx += 1
        else:
            idx += 1

    return SIVC


def plot_S_and_lnS(S_mean, S_error, q_i):
    itau = np.arange(len(S_mean))
    S = S_mean[:, q_i]
    S_err = S_error[:, q_i]

    mask = S > 0
    itau_p = itau[mask]
    S_p = S[mask]
    S_err_p = S_err[mask]
    lnS_p = np.log(S_p)
    lnS_err_p = S_err_p / S_p


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].errorbar(itau_p, S_p, yerr=S_err_p)
    axes[0].set_xlabel("itau")
    axes[0].set_ylabel("S")
    axes[0].set_title(f"S vs itau (q={q_i})")

    axes[1].errorbar(itau_p, lnS_p, yerr=0)
    axes[1].set_xlabel("itau")
    axes[1].set_ylabel("ln S")
    axes[1].set_title(f"ln S vs itau (q={q_i})")

    plt.tight_layout()
    plt.savefig(f"./re5000_L6/plot/S_lnS_q{q_i}.png")
    plt.close()

def compute_S_mean_and_S_error(SIVC):

    size, ntau, n_dot = SIVC.shape
    
    S_mean = np.zeros((ntau, n_dot), dtype=float)
    S_error = np.zeros((ntau, n_dot), dtype=float)
    

    for itau in range(ntau):
        for iq in range(n_dot):
            values = SIVC[:, itau, iq]

            S_mean[itau, iq] = np.mean(values)

            S_error[itau, iq] = np.std(values)
    
    return S_mean, S_error


def plot_S_and_lnS(S_mean, S_error, q_i):
    itau = np.arange(len(S_mean))
    S = S_mean[:, q_i]
    S_err = S_error[:, q_i]

    mask = S > 0
    itau_p = itau[mask]
    S_p = S[mask]
    S_err_p = S_err[mask]
    lnS_p = np.log(S_p)
    lnS_err_p = S_err_p / S_p


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].errorbar(itau_p, S_p, yerr=S_err_p)
    axes[0].set_xlabel("itau")
    axes[0].set_ylabel("S")
    axes[0].set_title(f"S vs itau (q={q_i})")

    axes[1].errorbar(itau_p, lnS_p, yerr=0)
    axes[1].set_xlabel("itau")
    axes[1].set_ylabel("ln S")
    axes[1].set_title(f"ln S vs itau (q={q_i})")

    plt.tight_layout()
    plt.savefig(f"./bb_test/plot/S_lnS_q{q_i}.png")
    plt.close()


filename = "bb_test/results/FSIVCqt_allr.dat"
SIVC = read_FSIVCqt_auto(filename)
S_mean, S_error = compute_S_mean_and_S_error(SIVC)

for i in range(9):
    plot_S_and_lnS(S_mean, S_error, i)