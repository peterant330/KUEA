import sys
import open_clip
import torch
import torch.nn as nn
class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, projector):
        super().__init__()
        self.model = model
        self.projector = projector

    def forward(self, vision):
        embedding = self.model(vision)
        embedding = self.projector(embedding)
        return embedding
def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu", lora=False):
    if model_name == "ViT-L-14-DIVA":
        import clip
        model, transform = clip.load('ViT-L/14', "cpu")
        model.load_state_dict(torch.load("../checkpoint/OpenAI-ViT-L-14-224.pth", map_location=torch.device('cpu')))
        model.eval()
        model = model.to(device)
        tokenizer = clip.tokenize
        return model, transform, tokenizer
    elif model_name == "ViT-L-14-336-DIVA":
        import clip
        model, transform = clip.load('ViT-L/14@336px', "cpu")
        model.load_state_dict(torch.load("../checkpoint/OpenAI-ViT-L-14-336.pth", map_location=torch.device('cpu')))
        model.eval()
        model = model.to(device)
        tokenizer = clip.tokenize
        return model, transform, tokenizer
    elif model_name == "SigLip":
        model, transform = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-SO400M-14-SigLIP', device='cpu'
        )
        if pretrained != 'openai':
            if isinstance(pretrained, str):
                checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
            else:
                checkpoint = pretrained
            if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            else:
                model.visual.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP')
        return model, transform, tokenizer
    elif model_name == "DFN":
        model, transform = open_clip.create_model_from_pretrained(
            'hf-hub:apple/DFN2B-CLIP-ViT-L-14', device='cpu'
        )
        if pretrained != 'openai':
            if isinstance(pretrained, str):
                checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
            else:
                checkpoint = pretrained
            if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            else:
                model.visual.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        return model, transform, tokenizer
    elif model_name == "MetaCLIP":
        model, transform = open_clip.create_model_from_pretrained('ViT-L-14-quickgelu',
                                                                  pretrained="metaclip_400m",
                                                                  device='cpu')
        if pretrained != 'openai':
            if isinstance(pretrained, str):
                checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
            else:
                checkpoint = pretrained
            if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            else:
                model.visual.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        return model, transform, tokenizer
    try:
        model, transform = open_clip.create_model_from_pretrained(
            model_name, pretrained='openai', cache_dir=cache_dir, device='cpu'
        )
        if lora:
            original_named_modules = model.named_modules()
            name_list = []
            for name, module in original_named_modules:
                if isinstance(module, nn.MultiheadAttention):
                    name_list.append(name)
            for name, module in model.named_modules():
                if name not in name_list:
                    continue
                parent_name = '.'.join(name.split('.')[:-1])  # Get the parent module name
                attr_name = name.split('.')[-1]  # Get the last part of the name (the actual module name)

                # Use getattr to access the parent module and set the new layer
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)

                # Replace the original module with the new one
                setattr(parent_module, attr_name,
                        PlainMultiheadAttentionLoRA(module, enable_lora=['q', 'k', 'v'], r=8, lora_alpha=16,
                                                    dropout_rate=0.1))
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
        else:
            checkpoint = pretrained
        if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
            model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
        else:
            model.visual.load_state_dict(checkpoint)
    except Exception as e:
        # try loading whole model
        print(f'error: {e}', file=sys.stderr)
        print('retrying by loading whole model..', file=sys.stderr)
        torch.cuda.empty_cache()
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir, device='cpu'
        )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
