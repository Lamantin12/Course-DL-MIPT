import numpy as np

class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:
        output = module.forward(input)
    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.
        gradInput = module.backward(input, gradOutput)
    """
    def __init__(self):
        self.output = None
        self.grad_input = None
        self.training = True

    def forward(self, inp):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.update_output(inp)

    def backward(self, inp, grad_output):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.update_grad_input(inp, grad_output)
        self.acc_grad_params(inp, grad_output)
        return self.grad_input

    def update_output(self, inp):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.
        Make sure to both store the data in `output` field and return it.
        """
        # The easiest case:
        # self.output = input
        # return self.output
        pass

    def update_grad_input(self, inp, grad_output):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.
        The shape of `gradInput` is always the same as the shape of `input`.
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        # The easiest case:
        # self.gradInput = gradOutput
        # return self.gradInput
        pass
    
    def acc_grad_parameters(self, inp, grad_output):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zero_grad_parameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def get_parameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def get_grad_parameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"

class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.
         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def update_output(self, inp):
        """
        Basic workflow of FORWARD PASS:
            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})
        Just write a little loop.
        """
        self.output = inp
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output

    def backward(self, inp, grad_output):
        """
        Workflow of BACKWARD PASS:
            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)
        !!!
        To ech module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.
        !!!
        """
        for i in range(len(self.modules) - 1, 0, -1):
            grad_output = self.modules[i].backward(self.modules[i - 1].output, grad_output)
        self.grad_input = self.modules[0].backward(inp, grad_output)
        return self.grad_input

    def zero_grad_parameters(self):
        for module in self.modules:
            module.zero_grad_parameters
    
    def get_grad_parameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.get_grad_parameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self,x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()

class Criterion(object):
    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, inp, target):
        """
            Given an input and a target, compute the loss function
            associated to the criterion and return the result.
            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.update_output(inp, target)

    def backward(self, inp, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result.
            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.update_grad_input(inp, target)

    def update_output(self, inp, output):
        """
        Function to override.
        """
        return self.output

    def update_grad_input(self, input, target):
        """
        Function to override.
        """
        return self.grad_input

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"