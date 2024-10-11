from parameters import *
import matplotlib.pyplot as plt

figsfolder = f'results/{corpus}/{LLM}/figs/ranks/'
os.makedirs(figsfolder,exist_ok=True)

fig,ax = plt.subplots(1)

RS_ranks_filename = f'ranks_RS_sub_length{sub_length}'
distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize)
RS = np.loadtxt(f'{distfolder}{RS_ranks_filename}.txt')
v,c = np.unique(RS,return_counts=True)
p = c / np.sum(c)
v = v / ( len(RS) - 1)
ax.plot(v,p,'-',label='RS')
# print(f'{v=}')

SR_ranks_filename = f'ranks_SR_sub_length{sub_length}'
SR = np.loadtxt(f'{distfolder}{SR_ranks_filename}.txt')
v,c = np.unique(SR,return_counts=True)
p = c / np.sum(c)
v = v / (len(SR) - 1)
ax.plot(v,p,'--',label='SR')

# ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('Rank/' + r'$(N_s-1)$')
ax.set_ylabel('Probability(Rank)')
ax.set_title(f'{Ns=}')

figname = f'{figsfolder}ranks_layer{layer_id}_sub_length{sub_length}.pdf'
fig.savefig(figname,bbox_inches='tight')

