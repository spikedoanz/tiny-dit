from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist

from vae import VITVAE
from image import draw

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist(fashion=False)
  """
  # normalizing breaks things for some reason?
  X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
  X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
  """
  model = VITVAE()
  opt = nn.optim.Adam(nn.state.get_parameters(model))

  @TinyJit
  @Tensor.train()
  def train():
    opt.zero_grad()
    batch = X_train[Tensor.randint(getenv("BS", 32), high=X_train.shape[0])]
    out, _mu, _logvar = model(batch)
    loss = ((out - batch)**2).sum() / 28*28*32
    loss.backward()
    opt.step()
    return loss

  @TinyJit
  @Tensor.test()
  def test():
    batch = X_test[Tensor.randint(getenv("BS", 32), high=X_test.shape[0])]
    out, _mu, _logvar = model(batch)
    return ((out - batch)**2).sum() / 28*28*32

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 1000))):
    loss = train().item()
    if i%100 == 0:
      test_acc = test().item()
    t.set_description(f"loss: {loss:6.2} test_acc: {test_acc:6.2f}")
    
  images, _,_ = model(X_test[:100])
  for i in range(10):
    draw(images[i][0], f"sample_{i}.png")
    

