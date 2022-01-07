# Surrogate gradients for BrainScaleS-2

**Important:** To run the code in this repository you need access to a neuromorphic BrainScaleS-2 system and the corresponding software stack.


## Dependencies

We have listed tested dependencies in `requirements.txt`.


## Setting up

To allow Python to import the custom dependencies you have to set up the PYTHONPATH by executing:

```Bash
export PYTHONPATH=$PWD/src/py:$PYTHONPATH
```


## Training

The two example scripts in this repository provide sensible defaults.
To start a training run simulated in software:

```Bash
cd experiments/mnist
python mnist.py --software-only --output my-results.h5
```

For more command line arguments, please refer to the experiment scripts.


## Inference

To load a previously trained network and perform inference with it:

```Bash
cd experiments/mnist
python mnist.py --software-only --load-weights my-results.h5 --epochs 0
```

