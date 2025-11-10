import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LogFormatter
from py_decoder import UFDecoder
from some_codes import toric_code, surface_code_non_periodic
from simulation_run import num_decoding_failures_batch


# Local helper that runs the batch decoder but forces Algorithm 1 (via the new Python binding)
def num_decoding_failures_batch_alg1(decoder, logicals, p_err, p_erase, num_rep, topological=False):
    """Like simulation_run.num_decoding_failures_batch but uses decoder.decode_batch_alg1 when topological=True.
    Falls back to the original function for non-topological decoders.
    """
    if not topological:
        return num_decoding_failures_batch(decoder, logicals, p_err, p_erase, num_rep, topological=False)

    H = decoder.h
    n_syndr = H.shape[0]
    n_qbt = H.shape[1]
    syndrome = np.zeros(n_syndr * num_rep, dtype=np.uint8)
    erasure = np.zeros(n_qbt * num_rep, dtype=np.uint8)
    l_noise = []
    for i in range(num_rep):
        error_pauli = np.random.binomial(1, p_err, n_qbt).astype(np.uint8)
        erasure[i*n_qbt:(i+1)*n_qbt] = np.random.binomial(1, p_erase, n_qbt).astype(np.uint8)
        noise = np.logical_or(np.logical_and(np.logical_not(erasure[i*n_qbt:(i+1)*n_qbt]), error_pauli), np.logical_and(erasure[i*n_qbt:(i+1)*n_qbt], np.random.binomial(1, 0.5, n_qbt)))
        l_noise.append(noise)
        syndrome[i*n_syndr:(i+1)*n_syndr] = (H @ noise % 2).astype(np.uint8)

    # decode batch using Algorithm 1 binding
    decoder.correction = np.zeros(n_qbt * num_rep, dtype=np.uint8)
    decoder.decode_batch_alg1(syndrome, erasure, num_rep)

    # evaluate decoding
    n_err = 0
    for i in range(num_rep):
        error = (l_noise[i] + decoder.correction[i*n_qbt:(i+1)*n_qbt]) % 2
        if np.any(error @ logicals.T % 2):
            n_err += 1
    return n_err


########### simulate 2d surface code (with and without periodic boundaries): ###########
l_len = [10, 20, 40, 80]  # surface code distances
num_trials = 50000  # repetitions for averaging
p_erasure = 0.0  # erasure rates
a_pauli_error_rate = np.linspace(0.005, 0.10, num=20)  # Pauli error rates

# 2d surface code with non-periodic boundaries
for i, l in enumerate(l_len):
    H, logical = surface_code_non_periodic(l)  # create parity-check matrix and logicals
    uf_decoder = UFDecoder(H)  # setup decoder
    l_logical_error_rate = []
    for p_err in a_pauli_error_rate:
        num_err = num_decoding_failures_batch_alg1(uf_decoder, logical, p_err, p_erasure, num_trials, topological=True)
        l_logical_error_rate.append(num_err / num_trials)
    # mask zeros so they don't produce -inf on the log axis and create plotting artifacts
    y = np.array(l_logical_error_rate, dtype=float)
    # empirical zero rates (no failures observed). Floor them to 1/num_trials so they are visible on the log axis
    min_rate = 1.0 / float(num_trials)
    zero_mask = (y <= 0)
    # create a plotted array that replaces exact zeros with the floor so lines connect
    y_plot = y.copy()
    y_plot[zero_mask] = min_rate
    # add descriptive label for legend (line). We'll overlay special markers for floored points.
    plt.semilogy(a_pauli_error_rate, y_plot, 'ro-', alpha=0.4+0.6*((i+1)/len(l_len)), label=f"Surface (open), L={l}")
    # mark floored (zero) points so the reader knows these were empirical zeros (<=1/num_trials)
    if np.any(zero_mask):
        plt.semilogy(np.array(a_pauli_error_rate)[zero_mask], y_plot[zero_mask], 'rs', markerfacecolor='none', markersize=6,
                     label=f"<=1/{num_trials} (no failures)")

# 2d toric code with periodic boundaries
for i, l in enumerate(l_len):
    H, logical = toric_code(l)  # create parity-check matrix and logicals
    uf_decoder = UFDecoder(H)  # setup decoder
    l_logical_error_rate = []
    for p_err in a_pauli_error_rate:
        num_err = num_decoding_failures_batch_alg1(uf_decoder, logical, p_err, p_erasure, num_trials, topological=True)
        l_logical_error_rate.append(num_err / num_trials)
    # mask zeros so they don't produce -inf on the log axis and create plotting artifacts
    y = np.array(l_logical_error_rate, dtype=float)
    min_rate = 1.0 / float(num_trials)
    zero_mask = (y <= 0)
    y_plot = y.copy()
    y_plot[zero_mask] = min_rate
    # add descriptive label for legend (line). We'll overlay special markers for floored points.
    plt.semilogy(a_pauli_error_rate, y_plot, 'bo:', alpha=0.4+0.6*((i+1)/len(l_len)), label=f"Toric (periodic), L={l}")
    if np.any(zero_mask):
        plt.semilogy(np.array(a_pauli_error_rate)[zero_mask], y_plot[zero_mask], 'bs', markerfacecolor='none', markersize=6,
                     label=f"<=1/{num_trials} (no failures)")

# add axis labels, title, grid and legend
plt.xlabel('Pauli error rate')
plt.ylabel('Logical error rate')
plt.title('Logical error rate vs Pauli error rate')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# ensure the y-axis includes 1e-5 and uses decade ticks
ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(LogFormatter(base=10.0))
# set a sensible lower bound so 1e-5 appears; keep the upper bound autoscaled
ax.set_ylim(bottom=1e-5)
# force decade ticks including 1e-5
ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
# deduplicate legend entries (keep first occurrence) and keep the same title/style
handles, labels = ax.get_legend_handles_labels()
by_label = {}
for h, l in zip(handles, labels):
    if l not in by_label:
        by_label[l] = h
ax.legend(by_label.values(), by_label.keys(), title='Code type / parameters', loc='best', fontsize='small')
plt.show()


# save data in a file so I can make the graph better.