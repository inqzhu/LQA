# _*_ coding: utf-8 _*_
'''
LQA 寻找当前最优学习率方法
'''
import copy
import torch
import re

# gpu cuda 设备
_device = torch.device("cuda")
# 是否已经进行过变量名翻译
_is_dict = False
# 变量名称字典
_trans_name_dict = None

def translate_var_name(model):
    # 将变量名称转换成能够提取梯度的名称
    num_pat = re.compile(r'\.([0-9]+)\.')
    trans_name = {}
    for ws in model.state_dict():
        #print(ws)
        nums = num_pat.findall(ws)
        new_name = ws
        if len(nums)>0:
            for i in range(len(nums)):
                th = '[%s].' % nums[i]
                new_name = re.sub(r'\.'+nums[i]+'\.', th, new_name)
        trans_name[ws] = new_name
    return trans_name

def LQA_delta(optimizer, model, criterion, xtrain, ytrain, epoch=None, loss_value=None):
    # 必要输入：
    # optimizer-优化器
    # model-网络模型
    # criterion-损失函数
    # xtrain、ytrain-当前batch数据

    global _device
    global _is_dict
    global _trans_name_dict
        
    
    try:
        # 若还未进行过变量字典翻译
        if not _is_dict:
            _trans_name_dict = translate_var_name(model)
            
        # 若没有在输入中给出当前的loss value, 则补充一次计算
        if not loss_value:
            _output = model(xtrain)
            loss = criterion(_output, ytrain)
            loss_value = loss.detach().cpu().item()

        # 设置 delta_try
        if not epoch:
            delta_try = 1
        else:
            delta_try = 1 * max(1-epoch*0.02, 0.1)

        # 1-copy原始参数
        original_pars = copy.deepcopy(model.state_dict())

        # 2-计算 【参数 + delta_try * 梯度】
        temp_pars_plus = copy.deepcopy(original_pars)
        temp_pars_minus = copy.deepcopy(original_pars)
        temp_grads = {}
        for ws in original_pars:
            expr = 'temp_grads["%s"]=copy.deepcopy(model.%s.grad)' % (ws, _trans_name_dict[ws])
            # 获取当前梯度
            exec(expr, {'temp_grads': temp_grads, 'copy':copy, 'model': model})

        # * 调整梯度尺度
        max_step = -1
        for ws in original_pars:
            if temp_grads[ws] is not None:
                scale = max(temp_grads[ws].max(), abs(temp_grads[ws].min()))
                #temp_grads[ws] = temp_grads[ws] / scale
                if scale > max_step:
                    max_step = scale
        for ws in original_pars:
            if temp_grads[ws] is not None:
                temp_grads[ws] = temp_grads[ws] / max_step

        for ws in original_pars:
            if temp_grads[ws] is not None:
                temp_pars_plus[ws] = original_pars[ws].add(delta_try,temp_grads[ws])
                temp_pars_minus[ws] = original_pars[ws].add(-delta_try,temp_grads[ws])      

        # 计算两个尝试得到的 loss value
        with torch.no_grad():
            _tp_model = copy.deepcopy(model).to(_device)

            # 计算2个loss
            _tp_model.load_state_dict(temp_pars_plus)
            _tp_model.eval()
            output_plus = _tp_model(xtrain)
            loss_plus = criterion(output_plus, ytrain)
            loss_p_value = loss_plus.item()

            _tp_model.load_state_dict(temp_pars_minus)
            _tp_model.eval()
            output_minus = _tp_model(xtrain)
            loss_minus = criterion(output_minus, ytrain)
            loss_n_value = loss_minus.item()

        # 3-计算 optimal delta
        _e = 1e-24
        delta_opt = (loss_p_value - loss_n_value) * delta_try / 2 / (loss_p_value + loss_n_value - 2*loss_value+_e)
        delta_opt = min(abs(delta_opt), 0.5) 

        optimizer.param_groups[0]['lr'] = delta_opt
    except Exception as e:
        print("Error hapens in LQA: %s" % str(e))
        delta_opt = 0.01
    
    return delta_opt