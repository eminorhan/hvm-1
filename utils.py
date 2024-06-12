"""
Util functions
"""
import torch
import models_mae, models_vit
from huggingface_hub import hf_hub_download

def get_available_models():
    available_models = [
        'mae_hvm1_none',
        'vit_hvm1_none',
        'vit_hvm1_ssv2-50shot',
        'vit_hvm1_ssv2-10shot',
        'vit_hvm1_kinetics-50shot',
        'vit_hvm1_kinetics-10shot',
        'vit_hvm1_imagenet-2pt',
        'mae_hvm1@448_none',
        'vit_hvm1@448_none',
        'vit_hvm1@448_ssv2-50shot',
        'vit_hvm1@448_ssv2-10shot',
        'vit_hvm1@448_kinetics-50shot',
        'vit_hvm1@448_kinetics-10shot',
        'vit_hvm1@448_imagenet-2pt',
        'mae_kinetics_none',
        'vit_kinetics_none',
        'vit_kinetics_ssv2-50shot',
        'vit_kinetics_ssv2-10shot',
        'vit_kinetics_kinetics-50shot',
        'vit_kinetics_kinetics-10shot',
        'vit_kinetics_imagenet-2pt'
        ]

    return available_models

def load_model(model_name):

    # make sure model is valid
    assert model_name in get_available_models(), 'Unrecognized model!'

    # parse identifier
    model_type, pretrain_data, finetune_data = model_name.split('_')

    # checks
    assert model_type in ['mae', 'vit'], 'Unrecognized model type!'
    assert pretrain_data in ['hvm1', 'hvm1@448', 'kinetics'], 'Unrecognized pretraining data!'
    assert finetune_data in ['none', 'ssv2-50shot', 'kinetics-50shot', 'ssv2-10shot', 'kinetics-10shot', 'imagenet-2pt'], 'Unrecognized finetuning data!'

    # download checkpoint from hf
    ckpt_filename = pretrain_data + '_' + finetune_data + '.pth'
    ckpt = hf_hub_download(repo_id='eminorhan/hvm-1', filename=ckpt_filename)

    if pretrain_data == 'hvm1@448':
        img_size = 448
    else:
        img_size = 224

    if model_type.startswith('mae'):
        model = models_mae.mae_vit_huge_patch14(img_size=img_size)
        ckpt = torch.load(ckpt, map_location='cpu')
        msg = model.load_state_dict(ckpt['model'], strict=True)
        print(f'Loaded with message: {msg}')
    elif model_type.startswith('vit'):
        if finetune_data.startswith('ssv2'):
            num_classes = 174
        elif finetune_data.startswith('kinetics'):
            num_classes = 700
        elif finetune_data.startswith('imagenet'):
            num_classes = 1000
        else:
            num_classes = None
        model = models_vit.vit_huge_patch14(img_size=img_size, num_classes=num_classes)
        ckpt = torch.load(ckpt, map_location='cpu')['model']
        msg = model.load_state_dict(ckpt, strict=False)
        print(f'Loaded with message: {msg}')

    return model