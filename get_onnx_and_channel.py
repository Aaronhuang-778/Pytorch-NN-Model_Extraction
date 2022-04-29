import json
import torch

# net为待转化的网络；x为输入的张量，只需要维度符合即可；path为输出文件路径
def get_onnx_and_channel(net, x, onnx_path, channel_path):

    dm = net._modules

    # 获取channel
    channels_dict = {}
    index = 0
    for name, layer in dm.items():
        # func = FuncClassifier(name)
        from collections.abc import Iterable
        if isinstance(layer, Iterable):
            for i in layer:
                objlist = dir(i)
                if 'in_channels' in objlist:
                    dic = {
                        'in_channels': i.in_channels,
                        'out_channels': i.out_channels
                    }
                    channels_dict[index] = dic
                else:
                    # error: has not attribute
                    pass
                index += 1
        else:
            objlist = dir(layer)
            if 'in_channels' in objlist:
                dic = {
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels
                }
                channels_dict[index] = dic
            else:
                # error: has not attribute
                pass
            index += 1
    with open(channel_path, "w") as j:
        json.dump(channels_dict, j)
        
    params = {}
    for name, para in net.named_parameters():
        print(name, ":", para.size())
        params[name] = para.detach().cpu().numpy()
        params[name].tofile(name + ".bin")
    # 导出onnx
    # x = torch.rand(size=(8, 3, 224, 224))
    torch.onnx.export(net,
                      args=x,
                      f=onnx_path,
                      export_params=False,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}}
                      )