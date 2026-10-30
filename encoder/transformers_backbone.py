"""Shared Transformers backend for legacy vision encoders."""

def build(self, folder, meta, metadata, pretrained, with_decoder, method, size):
    from transformers import AutoConfig, AutoModel, SiglipVisionConfig, SiglipVisionModel
    config_dict = meta.get('model_config')
    if method == 'SigLIP2':
        # Fixed-resolution SigLIP2 checkpoints use the Siglip architecture;
        # the separate Siglip2 classes are for NaFlex checkpoints.
        config = (SiglipVisionConfig(**config_dict) if config_dict else
                  AutoConfig.from_pretrained(folder, local_files_only=True).vision_config)
        self.model = (SiglipVisionModel.from_pretrained(folder, local_files_only=True) if pretrained
                      else SiglipVisionModel(config))
        self.hw = (224, 224)
        self.mean, self.std = [.5]*3, [.5]*3
    else:
        config = (AutoConfig.for_model(**config_dict) if config_dict else
                  AutoConfig.from_pretrained(folder, local_files_only=True))
        self.model = (AutoModel.from_pretrained(folder, local_files_only=True) if pretrained
                      else AutoModel.from_config(config))
        if method == 'V-JEPA2':
            self.model.predictor = None
        self.hw = (256, 256) if method == 'V-JEPA2' else (224, 224)
        self.mean, self.std = [.485, .456, .406], [.229, .224, .225]
    self.embed_dim = config.hidden_size
    meta['model_config'] = config.to_dict()
