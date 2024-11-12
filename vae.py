from tinygrad import Tensor, nn



# https://github.com/tinygrad/tinygrad/blob/773d5b60bfb83411993c7b499413e42e7fc43dbd/extra/models/transformer.py
class TransformerBlock:
  def __init__(self,
               embed_dim=8, 
               num_heads=2, 
               ff_dim=8, 
               prenorm=False, 
               act=lambda x: x.relu(), 
               dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim,embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim,embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim,embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x


class PatchEmbedding:
  def __init__(self, patch_size=2, in_channels=1, embed_dim=8):
    self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

  def __call__(self, x):
    return self.proj(x).rearrange("b c h w -> b (h w) c")

class VITVAE:
  def __init__(self,
               img_size=(28,28),
               patch_size=2,
               in_channels=1,
               embed_dim=28,
               enc_depth=4,
               dec_depth=4*2,
               latent_dim=8,
               num_heads=2):
    self.img_size = img_size 
    self.patch_size = patch_size
    self.grid_size = (img_size[0]//patch_size, img_size[1]//patch_size)
    self.img_size = img_size
    self.num_patches = self.grid_size[0] * self.grid_size[1]
    
    self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
    self.encoder = [TransformerBlock(embed_dim,num_heads) for _ in range(enc_depth)]
    self.to_latent = nn.Linear(embed_dim, 2*latent_dim)
    self.from_latent = nn.Linear(latent_dim, embed_dim)
    self.decoder = [TransformerBlock(embed_dim,num_heads) for _ in range(dec_depth)]
    self.final_proj = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
  
  def encode(self, x):
    x = self.patch_embed(x)
    for block in self.encoder:
      x = block(x)
    stats = self.to_latent(x)
    mu, logvar = stats.chunk(2, dim=-1)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = (logvar * 0.5).exp()
    eps = Tensor.randn(std.shape)
    z = mu + eps * std
    return z

  def decode(self, z):
    x = self.from_latent(z)
    for block in self.decoder:
      x = block(x)
    x = self.final_proj(x)
    x = x.rearrange(
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h = self.img_size[0]//self.patch_size,
        w = self.img_size[1]//self.patch_size,
        p1 = self.patch_size,
        p2 = self.patch_size,
      )
    return x

  def __call__(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z)
    return recon, mu, logvar


if __name__ == "__main__":
  x = Tensor.randn(2, 1, 28, 28)
  
  patch_embed = PatchEmbedding()
  pe_out = patch_embed(x)
  assert pe_out.shape == (2, 196, 8)
  
  transformer = TransformerBlock()
  t_out = transformer(pe_out)
  assert t_out.shape == (2, 196, 8)
  
  vae = VITVAE()
  mu, logvar = vae.encode(x)
  assert mu.shape == (2, 196, 8)
  assert logvar.shape == (2, 196, 8)
  
  z = vae.reparameterize(mu, logvar)
  assert z.shape == (2, 196, 8)
  
  dec = vae.decode(z)
  assert dec.shape == x.shape
  
  recon, mu, logvar = vae(x)
  assert recon.shape == x.shape
  assert mu.shape == (2, 196, 8)
  assert logvar.shape == (2, 196, 8)
