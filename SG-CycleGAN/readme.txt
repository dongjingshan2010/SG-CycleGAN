10.30日：train_conv_flow_dann 增加梯度反转训练域分类器，以获得模态无关特征。











train_conv_flow 共享网络采用卷积，而不是vit，vit运行太慢了。

同时集成进了rectified flow的训练和测试，

训练：MRI直接从共享网络提取的模态无关特征图映射到目标图像（MRI），
 测试：超声通过共享网络提取模态无关特征图，映射到目标图像（伪MRI）



train_vit_flow 共享网络采用vit



models_conv：共享层采用卷积编解码，输出模态无关特征是图像维度，3*H*W
models_vit:同上，共享层采用vit
models:不要求输出模态无关特征是图像维度



train_conv_flow 中的rectified flow 训练时是MRI提取的模态无关特征到真实MRI，但测试时使用MRI转模态无关再flow映射MRI为什么还是不清晰？说明程序有问题！