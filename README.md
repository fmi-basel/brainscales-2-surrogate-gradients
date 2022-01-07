# Surrogate gradients for BrainScaleS-2

Warning: To fully execute the code available within this repository you will need access to a neuromorphic BrainScaleS-2 system and the corresponding software stack.

You may explore the examples provided here in a software-only mode by specifying `--software-only`.


## Dependencies

We have listed tested dependencies in `requirements.txt`.


## Setting up

To allow Python to import the custom dependencies you have to set up the PYTHONPATH by executing:

```Bash
export PYTHONPATH=$PWD/src/py:$PYTHONPATH
```


## Training

The two exemplary experiment scripts already provide suitable defaults.
You may issue a training run by executing:

```Bash
cd mnist
python mnist.py --software-only --output my-results.h5
```

For more command line arguments, please refer to the experiment scripts.


## Inference

You may load previous training data by:

```Bash
cd mnist
python mnist.py --software-only --load-weights my-results.h5 --epochs 0
```

