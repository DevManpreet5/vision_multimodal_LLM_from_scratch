import torch
import torch.nn as nn

class siglipConfig:
    def __init__(self,hidden_size=768,intermediate_size=3072,num_hidden_layer=12,num_attentionhead=12,num_channel=3,image_size=224,patch_size=16,layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layer
        self.num_attention_heads = num_attentionhead
        self.num_channels = num_channel
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
    

class siglipVisionTransformer(nn.Module):
    def __init__(self,config):
        self.config=config
        embed_dim = config.hidden_size

        self.embedding=siglipEmbedding(config)
        self.encoder=siglipEncoder(config)
        self.post_layernorm=nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
    
    def forward(self,x):
        hidden_state=self.embedding(x)
        final=self.encoder(hidden_state)
        final=self.post_layernorm(hidden_state)

        return final
    
class SiglipVisionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = siglipVisionTransformer(config)

    def forward(self, pixel_values) :
        return self.vision_model(pixel_values=pixel_values) 
    

class siglipEmbedding(nn.Module):
    def __init__(self,config):
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", 
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values):
        
        patch_embeds = self.patch_embedding(pixel_values)  
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


        