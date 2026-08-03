When we create checkpoints with pytorch, we have the models pickled side by side with the weights.
In inference, I want to create the model with the normal python constructor and load the weights.

So:

Case 1:

```python
model = torch.load('model.pt')
```

Case 2:

```python

model = MyModelClass()
weights = torch.load('weights.pt')
model.load_state_dict(weights)
```


The way the code is currently prevent us from doing Case 2. This is because we use Hydra when training the model, and Hydra is used in many of the __init__ method of many modules in that package using:

```python

from hydra.utils import instantiate

class Foo(torch.nn.Module):

     def __init__(self, some_options, ....):
         ...
         self.bar = instantiate(....)
```

The goal: recreate MyModel without Hydra.

My idea is first to change all "from hydra.utils import instantiate" to "from anemoi.model.utils import instantiate" and have the "instantiate" there call the Hydra one. That will be the first step, so we do not affect training.

Let's assume that we have a JSON or YAML file that contain a dictionnary ofthe resolved hydra config (a single object, all omegaconf substitutions resolved).

Second step would be to recreate the model from that dictionnary for inference. The instantiate will not forward to hydra, but we will instanciate object ourselves. It is pretty easy to implement.

It is understood that we will need a global variable somewhere that makes the difference between the training behaviour and the inference behaviour.

I want the change thought the code to be minimum. If my idea works, we only change a few imports everywhere, and then the bulk of the code happens in a single file.

Please comment, plan and implement. Before implementing, write a detailed markdown files with the ideas and plan.
