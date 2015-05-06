Cogngin v0.1 Design Document
author: ZE YU

============
  Contents
============
1. Phase explained
2. Neural network related design
3. Parallel/distributed implementation design

====================
  Phase Explained
====================
Phase 1, implement pure data-parallel neural network training
using Spark

======================
    Neural Network
======================
1. Scope
    This section introduces the design of the components that run *ONE*
    iteration at *ONE* single node, aka, the neural network itself, like how to
    define the structure, how to do forward/backward propagation, the error
    function, the gradient etc. All these are *NOT* related to Spark in
    data-parallel implementation.
    
    In other words, it's what happens on one worker node
    in the data-parallel architecture. Later however, if we also implement
    model-parallel, this could also be a reference as well.

2. Overview
    First, we design a single layer of the NN, which we call Module. Then we
    introduce how to combine Modules together. Finally, we design facilities
    that help training the model on one node. 

3. Module (trait)
    A Module represents a single layer of the NN. A Module has the following
    fundamental factors:
    - the data operation within the layer, including (1) forward function, aka,
      given the input Xn-1, how to compute the output Xn, (2) backward
      propagation, given the gradient back-propagated from layer n+1, aka, dE/dXn,
      how to calculate the gradient against its input, dE/dXn-1, and the gradient
      against its own parameter, dE/dWn

    - how does it connect to the next layer, aka, the dependency.
      For now I think fully connected Modules are enough, but later we might 
      need other dependency like convolution network.

    The activation function can be implemented by using some existing code, like
    sigmoid, linear or softmax.

    The Module trait should be implemented by some concrete Module, like linear
    Module, Softmax or output layer. The exact instance of Module should be
    created by using a factory function. 

4. Neural Network 
    An instance of NN defines the model consist of multiple Modules. This 
    should be done by calling connectTo() function of the Module

    We should also be able to define error function that computes error value 
    given an example's label and the output from the last module (output layer)

    The most important behavior of NN is to call the train() function, which
    takes a set of learning examples as input and pull the trigger to learn.
    Conceptually, it is the for loop that iterates over the input set, runs
    forward and backward, accumulate local model update. Moreover, this is the
    only component that is related to Spark, since it defines how to *GLOBALLY*
    update the model.

5. Other facilities
    Other facilities should be defined as needed. For example, we might need to
    define functions to initialize parameters etc. Put them in the appropriate
    class, if it doesn't belong to anything, put it into the utils package
