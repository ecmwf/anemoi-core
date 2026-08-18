I want to implement object injection of embedded object in all the classes that call hydra's "instanciate". This is a big job.
All objects contain polymorphic sub-objecst must have sub-obects passed as a paramterer to the contructor. So

```python

class Foo(nn.Module):

    def __init__(self, ...):

        self.processor = instanciate(...)

```

Becomes

```python

class Foo(nn.Module):

    def __init__(self, processor, ...):

        self.processor = processor

```

I want you to implement a "Builder" class that that a dictionnary (i.e. the hydra config), and create the network on object containment that we have now, but only using object injection. The change apply to all three subpackages: models, graphs and training. Common code should go to anemoi-utils if needed.

Do not just substitite hydra to a similar class or function:

1 - I want all the objects to recieve their member object ("has-a" relationship) as full build objects as parameteres to their constructors

2 -  I want some code (build) that read the hydra config, and then build the models/graphs/etc using point 1).

Read alos graph.md
