#!/usr/bin/python
from functions import * 

LLM = 'OPT'
model,config,tokenizer = get_model(LLM)

# elif LLM=='Pythia':
#   from transformers import (GPTNeoXForCausalLM,
#                             GPTNeoXConfig
#                             )
#   modelname = "EleutherAI/pythia-410m-deduped"
#   model = GPTNeoXForCausalLM.from_pretrained(
#                             modelname,
#                             revision="main",
#                             # cache_dir="./pythia-410m-deduped/main",
#                             )#.to(device)
#   config = GPTNeoXConfig.from_pretrained(modelname,
#                                        # output_hidden_states=True,
#                                         )
# with open("models.json",'r') as f:
#     model_path = json.load(f)[modelname]

# if modelname.split('')[0]=='pythia':
#     from transformers import GPTNeoXForCausalLM as MOD

# elif modelname.split('')[0]=='olmo':
#     from hf_olmo import OLMoForCausalLM as MOD

# else:
#     from transformers import AutoModelForCausalLM as MOD

# md = MOD.from_pretrained(model_path)#,torch_dtype=torch.float16)

output = dict()

# print(f'{model=}')
# print('')
print(f'{model.base_model.decoder.layers=}')
num_heads = model.config.num_attention_heads
print(f'{num_heads=}')
head_dim = model.config.hidden_size // num_heads  # Hidden size per head
print(f'{head_dim=}')
for layer_id,l in enumerate(model.base_model.decoder.layers):
  # print(f'{i=}')
  output[layer_id] = {'self_attn_layernorm':
                  {'weight': l.self_attn_layer_norm.weight.detach().numpy(),
                    'bias': l.self_attn_layer_norm.bias.detach().numpy()
                  },
                'final_layer_norm':
                  {'weight': l.final_layer_norm.weight.detach().numpy(),
                    'bias': l.final_layer_norm.bias.detach().numpy()
                  },
                  'Q':l.self_attn.q_proj.weight.view(num_heads, head_dim, -1).detach().numpy(),
                  'K':l.self_attn.k_proj.weight.view(num_heads, head_dim, -1).detach().numpy(),
                  'V':l.self_attn.v_proj.weight.view(num_heads, head_dim, -1).detach().numpy(),
                  'fc1':
                  {'weight':l.fc1.weight.detach().numpy(),
                  'bias':l.fc1.bias.detach().numpy()
                  },
                  'fc2':
                  {'weight':l.fc2.weight.detach().numpy(),
                   'bias':l.fc2.bias.detach().numpy()
                   },
  }
                  
print(f'{output[layer_id]["fc1"]["weight"].shape=}')
print(f'{output[layer_id]["fc1"]["bias"].shape=}')

layer_idx = 27
f = sys.argv[1]
print(f'{f=}')

resultsfolder = get_weights_folder(LLM,layer_idx,layer_name=f)
extract_MLP(output,layer_idx,f,resultsfolder)

