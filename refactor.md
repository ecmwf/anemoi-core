I want now do get rid of "instanciate". Let' also remove hydra for now. We will bring it back later.

Let's start with "models" and "graphs". When creating a model, want an abstract Config object to be passed to all constructors.

```python
class Foo(nn.Module):
   def __init__(self, config: Config, ...):
       ...
```


The rule above applies to all Module that instanciate further Module with instanciate. For now, we will leave other Module (e.g. FFT2D layer)

We want the end-user to create models and graphs directly in Python.

```python
model = MyModel(config, more_stuff)
```

If a module calls instanciate now, we expect the user to pass
and object directly.

```python
model = MyModel(config, some_layer=MyLayer(config, ...), ...)
```

My default, some_layer will be None, in that case MyModel will build use the default class. If some_layer is a string, call `config.create_module(some_layer)`, that will help later when we bring back hydra.

The Config is and ABC because we will use a different subclass
when building models in training and in inference.


Values are read from the config with the "get()" abstract method.

```
value = config.get("something", default=42)
```

If the key does not exists and no default is given, raise error.

In training, Config will be created from the dataset, so values like "number_of_channels" are extracted from the training dataset, and the model is create accordingly.

The Config must be serialisable in JSON.

In inference, Config will be created from the JSON of a serialised training Config.


```Python

class MyClass(nn.Module):

    def __init__(self, config, my_sub_module = None):

        match my_sub_module:
            case None:
                self.my_sub_module = MyDefaultSubModule(config)
            case str():
                self.my_sub_module = config.create_module(my_sub_module)
            case _:
                self.my_sub_module = my_sub_module
```

Capture our conversation and descisions in a file. Describe the options we did not select as well. Update/crate/delete skills as required.

Implement the JSON based Config, and implement some code that build a model, prefereably one from the existing examples.

Let's rename "Config" to "Parametrisation" for now, and the variables "params".

Move the Parametrisation to anemoi-utils, as well as the concrete DictParametrisation (the one we can recreate from a JSON file). I have cloned anemoi-utils in the current directory.

The other subclass of Parametrisation will live in training, and will be called something like TrainingParametrisation


Change config=params in the methods paramter lists.
We should not have a "build" function. Everything goes though parametrisation as a paramater to __init__ and other methods. That inclused graphs.
