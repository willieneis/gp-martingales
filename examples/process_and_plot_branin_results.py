import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_trace(yout):
    """Return BSF trace."""
    yout = np.array(yout).reshape(-1)
    yout_cum_min = np.minimum.accumulate(yout)
    return yout_cum_min



# Script to plot
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.figsize':[10., 6.]})
plt.figure()

methods = ['gplcb', 'cslcb']
seeds = range(1, 11)
x_iter = list(range(51))
h_list = []

for method in methods:
    traces = []

    for seed in seeds:
        filename = 'result_' + method + '_seed_' + str(seed) + '.pkl'
        data = pickle.load(open(filename, 'rb'))
        
        trace = get_trace(data.y)
        trace = trace[:len(x_iter)]
        traces.append(trace)

    
    traces_mean = np.mean(traces, 0)
    traces_std = np.std(traces, 0)
    traces_error = traces_std / np.sqrt(len(traces))
    
    h = plt.errorbar(x_iter, traces_mean, traces_error)
    h_list.append(h[0])

# Plot optimal line
branin_min = 0.397887
h = plt.plot([0, 50], [branin_min, branin_min], '--')
h_list.append(h[0])

# Legend

# Plot settings
plt.xlabel('Iteration')
plt.ylabel('Minimum queried $f(x)$')
plt.ylim([0, 2.5])
plt.xlim([0, 50.])
plt.legend(h_list, ['GP-LCB', 'CS-LCB', 'Optimal $f*(x)$'])
plt.title('Branin')

filename_img = 'branin_plot.pdf'
plt.savefig(filename_img, bbox_inches='tight')
print('Saved figure: {}'.format(filename_img))
