from parameters import *
import matplotlib.pyplot as plt

figsfolder = f'results/{corpus}/{LLM}/figs/ranks/'
os.makedirs(figsfolder,exist_ok=True)

fig,ax = plt.subplots(1)
sub_lengths = [100]
for sub_length_id,sub_length in enumerate(sub_lengths):
  RS_ranks_filename = f'ranks_RS_sub_length{sub_length}'
  RS = np.loadtxt(f'{distfolder}{RS_ranks_filename}.txt')
  mean_RS = np.mean(RS) / N_batches / batch_size

  SR_ranks_filename = f'ranks_SR_sub_length{sub_length}'
  SR = np.loadtxt(f'{distfolder}{SR_ranks_filename}.txt')
  mean_SR = np.mean(SR) / N_batches / batch_size

# ax.set_xscale('log')
ax.set_yscale('log')
print(f'{mean_SR=}')
print(f'{mean_RS=}')
# figname = f'{figsfolder}ranks_layer{layer_id}_sub_length{sub_length}.pdf'
# fig.savefig(figname,bbox_inches='tight')

