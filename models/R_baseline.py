import torch
from torch import nn

class R_SpliterSimple(nn.Module):

    def __init__(self, max_delta_time, ways, emb_dim):
          super(R_SpliterSimple, self).__init__()
          ### New layers:
          self.emb_dim = emb_dim
          self.delta_time = max_delta_time
          self.delta_time_emb = nn.Embedding(max_delta_time, emb_dim)
          self.dir_emb = nn.Embedding(ways, emb_dim)
          self.lin_0  = nn.Linear(emb_dim, emb_dim)
          self.dropout = nn.Dropout(0.3)
          self.lin_1  = nn.Linear(80, 128)
          self.lin_2 = nn.Linear(128, 64)
          self.lin_3 = nn.Linear(64, 32)
          self.lin_4 = nn.Linear(32, 16)
          self.lin_5 = nn.Linear(16, 8)
          self.lin_6 = nn.Linear(8, 4)
          self.lin_7 = nn.Linear(4, 2)
          self.lin_8 = nn.Linear(2, 1)

    def forward(self, text_hidden_states, prior_embeds, delta_time, direction):
          delta_time_embedding = self.delta_time_emb(delta_time)
          dir_embedding = self.dir_emb(direction)
          pre_out = self.lin_0(prior_embeds)
          concat_data = torch.concat([text_hidden_states,
                                      pre_out,
                                      delta_time_embedding,
                                      dir_embedding
                                      ],
                                      axis = 1)
          #print(concat_data.shape)
          concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = 1)
          out = self.lin_1(concat_data.permute(0,2,1))
          out = nn.ELU()(out)
          out = self.lin_2(out)
          out = nn.ELU()(out)
          out = self.lin_3(out)
          out = nn.LayerNorm(out.shape[-1], elementwise_affine=False)(out)
          out = nn.ELU()(out)
          out = self.dropout(out)
          out = self.lin_4(out)
          out = nn.ELU()(out)
          out = self.lin_5(out)
          out = nn.LayerNorm(out.shape[-1], elementwise_affine=False)(out)
          out = nn.ELU()(out)
          out = self.dropout(out)
          out = self.lin_6(out)
          out = nn.ELU()(out)
          out = self.lin_7(out)
          out = self.dropout(out)
          out = nn.ELU()(out)
          out = self.lin_8(out).permute(0,2,1)
          return out