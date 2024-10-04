from parameters import *
import matplotlib.pyplot as plt

figsfolder = f'results/{corpus}/{LLM}/figs/ranks/'
os.makedirs(figsfolder,exist_ok=True)

fig,ax = plt.subplots(1)

RS_ranks_filename = f'ranks_RS_sub_length{sub_length}'
RS = np.loadtxt(f'{distfolder}{RS_ranks_filename}.txt')
v,c = np.unique(RS,return_counts=True)
p = c / np.sum(c)
v = v / N_batches / batch_size
ax.plot(v,p,'-',label='RS')

SR_ranks_filename = f'ranks_SR_sub_length{sub_length}'
SR = np.loadtxt(f'{distfolder}{SR_ranks_filename}.txt')
v,c = np.unique(SR,return_counts=True)
p = c / np.sum(c)
v = v / N_batches / batch_size
ax.plot(v,p,'--',label='SR')

# ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

figname = f'{figsfolder}ranks_layer{layer_id}_sub_length{sub_length}.pdf'
fig.savefig(figname,bbox_inches='tight')

