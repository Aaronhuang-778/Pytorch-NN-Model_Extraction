import onnx
from google.protobuf.json_format import MessageToDict, MessageToJson
import onnx_op
import json

id_dict = {}
ID = 0
index = 0


def onnx_input_analyze(_input):
    global ID, id_dict
    id_dict[_input[0]["name"]] = ID
    ID = ID + 1
    __input = onnx_op.Input('%' + str(id_dict[_input[0]["name"]]), _input[0])
    return __input


def onnx_output_analyze(_output):
    global ID, id_dict
    __output = onnx_op.Output('%' + str(id_dict[_output[0]["name"]]), '%' + str(ID) , _output)
    return __output


def onnx_nodes_analyze(nodes, channels):
    global ID, id_dict
    global index
    onnx_ops = []
    for node in nodes:
        if node['input'][0] not in id_dict:
            id_dict[node['input'][0]] = ID
            ID = ID + 1
        if node['output'][0] not in id_dict:
            id_dict[node['output'][0]] = ID
            ID = ID + 1

        if node['opType'] == 'Conv':
            onnx_ops.append(onnx_op.Conv2d('%' + str(id_dict[node['input'][0]]), '%' + str(id_dict[node['output'][0]]),
                                           node, channels[str(index)]))
        elif node['opType'] == 'Relu':
            onnx_ops.append(onnx_op.Relu('%' + str(id_dict[node['input'][0]]), '%' + str(id_dict[node['output'][0]])))
        elif node['opType'] == 'MaxPool':
            onnx_ops.append(
                onnx_op.MaxPool2d('%' + str(id_dict[node['input'][0]]), '%' + str(id_dict[node['output'][0]]),
                                  node))
        elif node['opType'] == 'AveragePool':
            onnx_ops.append(
                onnx_op.AveragePool2d('%' + str(id_dict[node['input'][0]]), '%' + str(id_dict[node['output'][0]]),
                                  node))
        elif node['opType'] == 'Flatten':
            onnx_ops.append(onnx_op.Flatten('%' + str(id_dict[node['input'][0]]), '%' + str(id_dict[node['output'][0]]),
                                            node))
        elif node['opType'] == 'Dense':
            onnx_ops.append(onnx_op.Dense('%' + str(id_dict[node['input'][0]]), '%' + str(id_dict[node['output'][0]]),
                                          node, channels[str(index)]))
        elif node['opType'] == 'Gemm':
            onnx_ops.append(onnx_op.Gemm('%' + str(id_dict[node['input'][0]]), '%' + str(id_dict[node['output'][0]]),
                                          node))
        else:
            print(node)

        index = index + 1

    return onnx_ops


if __name__ == "__main__":
    onnx_path = input("Please input the file's name of your onnx model:")
    onnx_channels_path = input("Please input the json's name of the channels of your onnx model:")
    onnx_model = None
    onnx_channels = None

    # 读取onnx文件和onnx channel文件，按照提示输入对应的路径即可
    try:
        onnx_model = onnx.load(onnx_path)
    except IOError:
        print("Illegal onnx path!")
    try:
        onnx_channels = open(onnx_channels_path, "r")
    except IOError:
        print("Illegal onnx channels path!")
    if onnx_model is None:
        exit("Wrong Onnx Model!")
    if onnx_channels is None:
        exit("Wrong Onnx Channels!")
    # onnx_json = MessageToJson(onnx_model)

    # 将两个onnx的文件内容转化为字典
    onnx_dict = MessageToDict(onnx_model)
    onnx_channels_dict = json.load(onnx_channels)

    # 分别处理输入，操作和输出
    onnx_nodes = onnx_dict['graph']['node']
    onnx_input = onnx_dict['graph']['input']
    onnx_output = onnx_dict['graph']['output']
    my_input = onnx_input_analyze(onnx_input)
    my_ops = onnx_nodes_analyze(onnx_nodes, onnx_channels_dict)
    my_output = onnx_output_analyze(onnx_output)

    # 从标准输入获取结果，也可以通过命令行重定向至文件
    print(my_input)
    for op in my_ops:
        print(op)
    print(my_output)

    onnx_channels.close()

# pyvgg16.onnx
# pyvgg16_channels.json
