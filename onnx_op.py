# 可以不用关心具体实现，总的说来就是解析字典然后字符串拼接

def search_attr(attr_list, obj: str):
    for attr in attr_list:
        if attr['name'] == obj:
            return attr
    return {'name': None}


def search_param(param_list, obj: str):
    for param in param_list:
        if obj in param:
            return param
    return None


class Input:
    def __init__(self, _output, body):
        self.output = _output
        self.shape = body['type']['tensorType']['shape']['dim']
        self.dtype = body['type']['tensorType']['elemType']

    def __str__(self):
        return self.output + '=input(' + ');' + str(self.shape) + str(self.dtype)


class Output:
    def __init__(self, _input, _output, body):
        self.input = _input
        self.output = _output
        self.body = body

    def __str__(self):
        return self.output + '=output(' + 'input=' + self.input + ');'


class Operator:
    def __init__(self):
        pass


class Conv2d(Operator):
    def __init__(self, _input, _output, body, channels):
        super().__init__()
        self.input = _input
        self.output = _output
        self.weight = search_param(body['input'], 'weight')
        self.bias = search_param(body['input'], 'bias')

        self.input_channels = channels['in_channels']
        self.output_channels = channels['out_channels']

        dilation = search_attr(body['attribute'], 'dilations')
        self.dilation = tuple([int(i) for i in dilation['ints']]) if dilation['name'] is not None else None
        kernel_size = search_attr(body['attribute'], 'kernel_shape')
        self.kernel_size = tuple([int(i) for i in kernel_size['ints']]) if kernel_size['name'] is not None else None
        stride = search_attr(body['attribute'], 'strides')
        self.stride = tuple([int(i) for i in stride['ints']]) if stride['name'] is not None else None
        padding = search_attr(body['attribute'], 'pads')
        self.padding = tuple([int(i) for i in padding['ints']]) if padding['name'] is not None else None

    def __str__(self):
        return self.output + '=nn.conv2d(' + 'input=' + self.input \
               + ',input_channel=' + str(self.input_channels) \
               + ',output_channel=' + str(self.output_channels) \
               + ',weight=' + str(self.weight) + '.bin' \
               + ',bias=' + str(self.bias) + '.bin' \
               + ',kernel_size=' + str(self.kernel_size) \
               + ',dilation=' + str(self.dilation) \
               + ',padding=' + str(self.padding) \
               + ',stride=' + str(self.stride) \
               + '); '


class Relu(Operator):
    def __init__(self, _input, _output):
        super().__init__()
        self.input = _input
        self.output = _output

    def __str__(self):
        return self.output + '=nn.relu(' + 'input=' + self.input + ');'


class MaxPool2d(Operator):
    def __init__(self, _input, _output, body):
        super().__init__()
        self.input = _input
        self.output = _output

        dilation = search_attr(body['attribute'], 'dilations')
        self.dilation = tuple([int(i) for i in dilation['ints']]) if dilation['name'] is not None else None
        kernel_size = search_attr(body['attribute'], 'kernel_shape')
        self.kernel_size = tuple([int(i) for i in kernel_size['ints']]) if kernel_size['name'] is not None else None
        stride = search_attr(body['attribute'], 'strides')
        self.stride = tuple([int(i) for i in stride['ints']]) if stride['name'] is not None else None
        padding = search_attr(body['attribute'], 'pads')
        self.padding = tuple([int(i) for i in padding['ints']]) if padding['name'] is not None else None

    def __str__(self):
        return self.output + '=nn.maxpool2d(' + 'input=' + self.input \
               + ',kernel_size=' + str(self.kernel_size) \
               + ',dilation=' + str(self.dilation) \
               + ',padding=' + str(self.padding) \
               + ',stride=' + str(self.stride) \
               + '); '


class AveragePool2d(Operator):
    def __init__(self, _input, _output, body):
        super().__init__()
        self.input = _input
        self.output = _output
        self.body = body

    def __str__(self):
        return self.output + '=nn.averagepool2d(' + 'input=' + self.input + ');'


class Flatten(Operator):
    def __init__(self, _input, _output, body):
        super().__init__()
        self.input = _input
        self.output = _output
        self.body = body

    def __str__(self):
        return self.output + '=nn.flatten(' + 'input=' + self.input + ');'


class Dense(Operator):
    def __init__(self, _input, _output, body, channels):
        super().__init__()
        self.input = _input
        self.output = _output
        self.input_channels = channels['in_channels']
        self.output_channels = channels['out_channels']

        self.weight = search_param(body['input'], 'weight')
        self.bias = search_param(body['input'], 'bias')

    def __str__(self):
        return self.output + '=nn.dense(' + 'input=' + self.input \
               + ',input_channel=' + str(self.input_channels) \
               + ',output_channel=' + str(self.output_channels) \
               + ',weight=' + str(self.weight) + '.bin' \
               + ',bias=' + str(self.bias) + '.bin' \
               + ');'


class Gemm(Operator):
    def __init__(self, _input, _output, body):
        super().__init__()
        self.input = _input
        self.output = _output
        self.weight = search_param(body['input'], 'weight')
        self.bias = search_param(body['input'], 'bias')

    def __str__(self):
        return self.output + '=nn.gemm(' + 'input=' + self.input \
               + ',weight=' + str(self.weight) + '.bin' \
               + ',bias=' + str(self.bias) + '.bin' \
               + ');'
