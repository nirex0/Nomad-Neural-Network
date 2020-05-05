# Nomad Neural Network
A Neural Network made in 10 lines using [Nomad](https://github.com/void-intelligence/Nomad) Matrix Library.

This repo here serves the purpose of an example project and nothing more, to see how you can work with Nomad to create a library of your own from scratch (or as scratch as it gets!)

## Library Quickstart

Here we'll go through a step by step guide on creating a Simple Network To learn XOR

First we need to install the library, so just do a simple 

```
PM> Install-Package NomadNetworkLibrary -Version 1.0.0
```

Now that we have the library, let's create our network.

```
var net = new DenseNet(new int[] { 3, 25, 25, 1 });
```

The network will automatically take care of the layer initialization based on the configuration array given to it as the constructor parameter.

We then create our inputs and outputs:

```
var inputs = new List<Matrix>();
var outputs = new List<Matrix>();

var m = new Matrix(3, 1);
var n = new Matrix(1, 1);

// 0 0 0    => 0
m[0, 0] = 0; m[1, 0] = 0; m[2, 0] = 0;
n[0, 0] = 0;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());

// 0 0 1    => 1
m[0, 0] = 0; m[1, 0] = 0; m[2, 0] = 1;
n[0, 0] = 1;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());

// 0 1 0    => 1
m[0, 0] = 0; m[1, 0] = 1; m[2, 0] = 0;
n[0, 0] = 1;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());

// 0 1 1    => 0
m[0, 0] = 0; m[1, 0] = 1; m[2, 0] = 1;
n[0, 0] = 0;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());

// 1 0 0    => 1
m[0, 0] = 1; m[1, 0] = 0; m[2, 0] = 0;
n[0, 0] = 1;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());

// 1 0 1    => 0
m[0, 0] = 1; m[1, 0] = 0; m[2, 0] = 1;
n[0, 0] = 0;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());

// 1 1 0    => 0
m[0, 0] = 1; m[1, 0] = 1; m[2, 0] = 0;
n[0, 0] = 0;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());

// 1 1 1    => 1
m[0, 0] = 1; m[1, 0] = 1; m[2, 0] = 1;
n[0, 0] = 1;
inputs.Add(m.Duplicate());
outputs.Add(n.Duplicate());
```

We then train the network using the FeedForward and Backprop methods in a loop:

```
for (var i = 0; i < 5000; i++)
{
    net.FeedForward(inputs[0]);
    net.BackProp(outputs[0]);

    net.FeedForward(inputs[1]);
    net.BackProp(outputs[1]);

    net.FeedForward(inputs[2]);
    net.BackProp(outputs[2]);

    net.FeedForward(inputs[3]);
    net.BackProp(outputs[3]);

    net.FeedForward(inputs[4]);
    net.BackProp(outputs[4]);

    net.FeedForward(inputs[5]);
    net.BackProp(outputs[5]);

    net.FeedForward(inputs[6]);
    net.BackProp(outputs[6]);

    net.FeedForward(inputs[7]);
    net.BackProp(outputs[7]);
}
```

After the network has been fully trained, we Test it and calculate the accuracy.

```
var correct = 0;
for (var i = 0; i < 125; i++)
{
    correct += Abs(net.FeedForward(inputs[0])[0, 0]) < 0.1 ? 1 : 0;
    correct += Abs(net.FeedForward(inputs[1])[0, 0]) - 1 < 0.1 ? 1 : 0;
    correct += Abs(net.FeedForward(inputs[2])[0, 0]) - 1 < 0.1 ? 1 : 0;
    correct += Abs(net.FeedForward(inputs[3])[0, 0]) < 0.1 ? 1 : 0;
    correct += Abs(net.FeedForward(inputs[4])[0, 0]) - 1 < 0.1 ? 1 : 0;
    correct += Abs(net.FeedForward(inputs[5])[0, 0]) < 0.1 ? 1 : 0;
    correct += Abs(net.FeedForward(inputs[6])[0, 0]) < 0.1 ? 1 : 0;
    correct += Abs(net.FeedForward(inputs[7])[0, 0]) - 1 < 0.1 ? 1 : 0;
}

var acc = (double)correct / 1000.0 * 100;
```

And we are done, this very example can also be found in the test project.

### Also!

About that 10 lines thing, if you actually go into the source code of the Layer function, you'll see that it operates with 10 lines of code (within functions of course)
