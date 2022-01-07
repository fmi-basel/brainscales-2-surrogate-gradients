import sys
import os
from datetime import datetime
import argparse
from collections import OrderedDict

import numpy as np
import h5py

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import strobe.nn
from strobe.datasets.mnist import MNIST16x16
from strobe.spikes import PixelsToSpikeTimes, SpikeTimesToDense
from strobe.datalogger import DataLogger

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--test-every", type=int, default=5)
parser.add_argument("--learning-rate", type=float, default=1.5e-3)
parser.add_argument("--learning-rate-decay", type=float, default=0.03)
parser.add_argument("--tau-input", type=float, default=8e-6)
parser.add_argument("--reg-bursts", type=float, default=0.0005)
parser.add_argument("--reg-weights-hidden", type=float, default=0.1)
parser.add_argument("--reg-readout", type=float, default=0.0)
parser.add_argument("--reg-weights-output", type=float, default=0)
parser.add_argument("--readout-scaling", type=float, default=10.0)
parser.add_argument("--jitter", type=float, default=0.0)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--output-group", type=str, default="results")
parser.add_argument("--load-weights", type=str, default=None)
parser.add_argument("--load-weights-group", type=str, default="results")
parser.add_argument("--load-weights-epoch", type=int, default=-1)
parser.add_argument("--save-test-traces", action="store_true")
parser.add_argument("--save-test-spikes", action="store_true")
parser.add_argument("--save-test-images", action="store_true")
parser.add_argument("--save-test-labels", action="store_true")
parser.add_argument("--random-rotation", type=float, default=0.0)
parser.add_argument("--software-only", action="store_true")
parser.add_argument("--n-samples", type=int, default=20)
parser.add_argument("--n-hidden", type=int, default=246)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--interpolation", type=int, default=1)
parser.add_argument("--calibration", type=str, default="cube_66.npz")
parser.add_argument("--inference-mode", action="store_true")

args = parser.parse_args()

if args.output is None:
    now = datetime.now()
    args.output = now.strftime("%d-%m-%Y-%H-%M-%S.h5")

    # raise only if the path was not explicitly given as an argument
    if os.path.exists(args.output):
        raise BaseException(f"File '{args.output}' already exists. Exiting.")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")

# fix seed
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(0, np.iinfo(np.int64).max, dtype=np.int64))


transform = transforms.Compose([
    transforms.RandomRotation(args.random_rotation),
    transforms.ToTensor(),
])

mnist_train = MNIST16x16("../../data/datasets", train=True, download=True, transform=transform)
mnist_test = MNIST16x16("../../data/datasets", train=False, download=True, transform=transforms.ToTensor())

n_input = 16*16
n_output = len(mnist_test.classes)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)

# set up input encoding and spike conversion
to_spike = PixelsToSpikeTimes(tau=args.tau_input, t_max=100e-6)
to_dense = SpikeTimesToDense(1.7e-6 / args.interpolation, args.n_samples*args.interpolation)

if not args.software_only:
    import pyhxcomm_vx as hxcomm
    connection = hxcomm.ManagedConnection().__enter__()
else:
    connection = None

# set the neuron parameters
# (leak and threshold are normalized to 0 and 1 respectively)
neuron_params = {
        "tau_mem": 6e-6,
        "tau_syn": 6e-6,
        }

model = strobe.nn.Network(OrderedDict([
    ("linear_hidden", strobe.nn.Linear(256, args.n_hidden, scale=240)),
    ("dropout_hidden", strobe.nn.Dropout(args.dropout, args.n_hidden)),
    ("hidden", strobe.nn.LIFLayer(args.n_hidden, neuron_params, activation_kwargs={"scale": 50.0})),
    ("linear_output", strobe.nn.Linear(args.n_hidden, 10, scale=240)),
    ("output", strobe.nn.LILayer(10, neuron_params))
    ]), interpolation=args.interpolation).to(device)
if not args.software_only:
    model.connect(
            connection=connection,
            calibration=args.calibration,
            sample_separation=200e-6 if not args.inference_mode else 11.8e-6,
            inference_mode=args.inference_mode
            )

# initialize or load weights
if args.load_weights is None:
    _time_step = 1.7e-6
    _tau_mem = 6e-6
    _beta = float(np.exp(-_time_step/_tau_mem))
    scale = 0.7 * (1.0 - _beta)
    torch.nn.init.normal_(model.linear_hidden.weight, mean=1.0e-3, std=scale / np.sqrt(n_input))
    torch.nn.init.normal_(model.linear_output.weight, mean=0.0e-3, std=scale / np.sqrt(args.n_hidden))
else:
    with h5py.File(args.load_weights, "r") as f:
        w_hidden = f[args.load_weights_group]["weight/hidden"][args.load_weights_epoch].T
        w_output = f[args.load_weights_group]["weight/output"][args.load_weights_epoch].T
        model.linear_hidden.weight.data = torch.tensor(w_hidden.astype(np.float32), requires_grad=True, device=device)
        model.linear_output.weight.data = torch.tensor(w_output.astype(np.float32), requires_grad=True, device=device)

    scale = min(63 / np.abs(model.linear_output.weight.detach().cpu()).max(), 250)
    model.linear_output.scale = scale

# set up opimizer and loss function
optimizer = Adam(model.parameters(), args.learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=1 - args.learning_rate_decay)

log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

datalogger = DataLogger(args.epochs + 1, len(train_loader))

for e in range(0, args.epochs + 1):
    if e > 0:
        for b, (x_local, y_local) in enumerate(train_loader):
            x_local, y_local = x_local.to(device), y_local.to(device)
            # set model to training mode and update dropout mask
            model.train()
            model.dropout_hidden.step()

            optimizer.zero_grad()

            # convert input data to spikes
            times = to_spike(x_local)
            if args.jitter > 0.0:
                times += torch.rand(times.shape) * args.jitter

            times -= times.min()

            spikes = to_dense(times)
            spikes = spikes.view((spikes.shape[0], spikes.shape[1], -1))

            # apply forward pass
            output = model(spikes)

            # add a tie breaker for software-only mode (fix depending on PyTorch version)
            output += torch.randn_like(output) * 1e-5

            # get max-over-time and extract classifiation response
            m, _ = torch.max(output, 1)
            _, am = torch.max(m, 1)

            m *= args.readout_scaling

            # regularization
            reg_bursts = args.reg_bursts * torch.mean(torch.sum(model.hidden.spikes, dim=1)**2)
            reg_weights_hidden = args.reg_weights_hidden * torch.mean(model.linear_hidden.weight**2)
            reg_weights_output = args.reg_weights_output * torch.mean(model.linear_output.weight**2)
            reg_readout = args.reg_readout * torch.mean(m**2)

            reg_loss = reg_bursts + reg_weights_hidden + reg_weights_output + reg_readout

            # calculate the loss and optimize
            log_p_y = log_softmax_fn(m)
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            loss_val.backward()
            optimizer.step()

            with torch.no_grad():
                limit = 63 / model.linear_hidden.scale
                model.linear_hidden.weight.data.clamp_(-limit, limit)

                scale = min(63 / np.abs(model.linear_output.weight.cpu().detach()).max(), 250)
                model.linear_output.scale = scale

            # calculate accuracy
            accuracy = np.mean((y_local == am).detach().cpu().numpy())
            rate = float(model.hidden.spikes.cpu().sum() / args.batch_size)
            sys.stdout.write(
                f"\r\033[Kbatch {b}: {loss_val:.3f} loss, {accuracy:.3f} accuracy, "
                + f"{rate:.1f} spikes per image")

            # store data
            datalogger.store("loss", float(loss_val.item()), e, b, average=(1, ))
            datalogger.store("accuracy/train", accuracy, e, b, average=(1, ))
            datalogger.store("rate/train", rate, e, b, average=(1, ))
            datalogger.store("regularizer/bursts", reg_bursts.item(), e, b, average=(1, ))
            datalogger.store("regularizer/weights/hidden", reg_weights_hidden.item(), e, b, average=(1, ))
            datalogger.store("regularizer/weights/output", reg_weights_output.item(), e, b, average=(1, ))

        datalogger.store("learning-rate", float(scheduler.get_last_lr()[0]), e)
        datalogger.store("weight/hidden", model.linear_hidden.weight.data.detach().cpu().numpy().T, e)
        datalogger.store("weight/output", model.linear_output.weight.data.detach().cpu().numpy().T, e)

        scheduler.step()

        sys.stdout.write(
            f"\r\033[Kafter {e} epochs: {datalogger.data['loss'][e].mean():.3f} loss, "
            + f"{datalogger.data['accuracy/train'][e].mean():.3f} train accuracy, "
            + f"{datalogger.data['rate/train'][e].mean():.1f} spikes per image\n")


    # calculate test accuracy
    if (args.test_every != -1 and e % args.test_every == 0) or (args.test_every == -1 and e == args.epochs):
        sys.stdout.write(f"\r\033[KApplying test dataset.")
        with torch.set_grad_enabled(False):
            model.eval()
            # assume batch size is equal to length of dataset
            x_local, y_local = next(iter(test_loader))
            x_local, y_local = x_local.to(device), y_local.to(device)

            times = to_spike(x_local)
            if args.jitter > 0.0:
                times += 0.5 * args.jitter

            times -= times.min()

            spikes = to_dense(times)
            spikes = spikes.view((spikes.shape[0], spikes.shape[1], -1))

            output = model(spikes)
            margin = 0.15 if not args.software_only else 0.0
            m, _ = torch.max(output - margin, 1)
            _, am = torch.max(m, 1)

            datalogger.store("accuracy/test", np.mean((y_local == am).detach().cpu().numpy()), e)
            datalogger.store("rate/test", (model.hidden.spikes.sum().cpu() /
                                           model.hidden.spikes.shape[0]).detach().numpy(), e)

            # in case we want to save images/spikes/traces, we should compress them
            compression_args = dict(compression=4)

            def convert(data, dtype):
                return data.detach().cpu().numpy().astype(dtype)

            if args.save_test_images:
                datalogger.store("images/test", convert(x_local, np.float16), e, h5_args=compression_args)

            if args.save_test_labels:
                datalogger.store("labels/test", convert(y_local, np.uint8), e, h5_args=compression_args)

            if args.save_test_spikes:
                datalogger.store("spikes/test/input", convert(spikes, np.bool), e, h5_args=compression_args)
                datalogger.store("spikes/test/hidden", convert(model.hidden.spikes, np.bool), e, h5_args=compression_args)

            if args.save_test_traces:
                datalogger.store("traces/test/hidden", convert(model.hidden.traces, np.float16), e, h5_args=compression_args)
                datalogger.store("traces/test/output", convert(model.output.traces, np.float16), e, h5_args=compression_args)

            sys.stdout.write(
                f"\r\033[Kafter {e} epochs: {datalogger.data['accuracy/test'][e]:.3f} test accuracy, "
                + f"{datalogger.data['rate/test'][e]:.1f} spikes per image\n")


with h5py.File(args.output, "a") as output:
    print(f"Saving data into {args.output} (group {args.output_group})")
    output_group = output.create_group(args.output_group)

    for k, v in args.__dict__.items():
        try:
            output_group.attrs[k] = v
        except TypeError:
            print(f"Could not save attribute {k}={v}")

    datalogger.dump(output_group)
