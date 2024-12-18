import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class CLIPTransform:
    def __init__(
        self,
        model_path='openai/clip-vit-large-patch14',
        device=None,
        dtype=None,
        local_files_only=True,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model_path = '/data3/models/huggingface/models--openai--clip-vit-large-patch14'
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.processor = None
        self.model = None

    def load_model(self):
        if self.model is None:
            processor = CLIPProcessor.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
            )
            model = CLIPModel.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
            )
            model.requires_grad_(False)
            if self.device is not None:
                model.to(self.device)
            if self.dtype is not None:
                model.to(self.dtype)
            self.processor = processor
            self.model = model

    @torch.no_grad()
    def __call__(self, text=None, image=None, text_w_proj=True, image_w_proj=True, to_numpy=True):
        # image: RGB
        self.load_model()
        results = []
        if text is not None:
            inputs = self.processor(
                text=text,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )
            input_ids = inputs.input_ids
            if self.device is not None:
                input_ids = input_ids.to(self.device)
            if text_w_proj:
                text_features = self.model.get_text_features(input_ids)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            else:
                text_features = self.model.text_model(input_ids)[1]
            if to_numpy:
                text_features = text_features.contiguous().cpu().numpy()
            results.append(text_features)
        if image is not None:
            inputs = self.processor(images=image, return_tensors='pt')
            pixel_values = inputs.pixel_values
            if self.device is not None:
                pixel_values = pixel_values.to(self.device)
            if self.dtype is not None:
                pixel_values = pixel_values.to(self.dtype)
            if image_w_proj:
                image_features = self.model.get_image_features(pixel_values)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            else:
                image_features = self.model.vision_model(pixel_values)[1]
            if to_numpy:
                image_features = image_features.contiguous().cpu().numpy()
            results.append(image_features)
        if len(results) > 1:
            return results
        elif len(results) == 1:
            return results[0]
        else:
            return None


class CLIPTextTransform:
    def __init__(
        self,
        model_path,
        device=None,
        dtype=None,
        revision=None,
        local_files_only=True,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.local_files_only = local_files_only
        self.tokenizer = None
        self.text_encoder = None
        self.load_model()

    def load_model(self):
        if self.text_encoder is None:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.model_path,
                subfolder='tokenizer',
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_path,
                subfolder='text_encoder',
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            text_encoder.requires_grad_(False)
            if self.device is not None:
                text_encoder.to(self.device)
            if self.dtype is not None:
                text_encoder.to(self.dtype)
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder

    @torch.no_grad()
    def __call__(self, text, mode, to_numpy=True):
        
        if mode == 'after_pool':
            inputs = self.tokenizer(
                text=text,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )
            input_ids = inputs.input_ids
            if self.device is not None:
                input_ids = input_ids.to(self.device)
            text_features = self.text_encoder(input_ids)[1]
        elif mode == 'before_pool':
            inputs = self.tokenizer(
                text=text,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            input_ids = inputs.input_ids
            if self.device is not None:
                input_ids = input_ids.to(self.device)
            text_features = self.text_encoder(input_ids)[0]
        else:
            assert False
        if to_numpy:
            text_features = text_features.contiguous().cpu().numpy()
        return text_features


class CLIPTextWithProjectionTransform:
    def __init__(
        self,
        model_path,
        device=None,
        dtype=None,
        revision=None,
        local_files_only=True,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.local_files_only = local_files_only
        self.tokenizers = None
        self.text_encoders = None

    def load_model(self):
        if self.text_encoders is None:
            tokenizer_one = CLIPTokenizer.from_pretrained(
                self.model_path,
                subfolder='tokenizer',
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            tokenizer_two = CLIPTokenizer.from_pretrained(
                self.model_path,
                subfolder='tokenizer_2',
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            text_encoder_one = CLIPTextModel.from_pretrained(
                self.model_path,
                subfolder='text_encoder',
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                self.model_path,
                subfolder='text_encoder_2',
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            if self.device is not None:
                text_encoder_one.to(self.device)
                text_encoder_two.to(self.device)
            if self.dtype is not None:
                text_encoder_one.to(self.dtype)
                text_encoder_two.to(self.dtype)
            self.tokenizers = [tokenizer_one, tokenizer_two]
            self.text_encoders = [text_encoder_one, text_encoder_two]

    @torch.no_grad()
    def __call__(self, text, mode, to_numpy=True):
        self.load_model()
        if mode == 'after_pool':
            text_features_list = []
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                inputs = tokenizer(
                    text=text,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                )
                input_ids = inputs.input_ids
                if self.device is not None:
                    input_ids = input_ids.to(self.device)
                text_features = text_encoder(input_ids, output_hidden_states=True)
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_text_features = text_features[0]
                text_features = text_features.hidden_states[-2]
                text_features = text_features[
                    torch.arange(text_features.shape[0], device=text_features.device),
                    input_ids.to(dtype=torch.int, device=text_features.device).argmax(dim=-1),
                ]
                text_features_list.append(text_features)
            text_features = torch.concat(text_features_list, dim=-1)
        elif mode == 'before_pool':
            text_features_list = []
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                inputs = tokenizer(
                    text=text,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                input_ids = inputs.input_ids
                if self.device is not None:
                    input_ids = input_ids.to(self.device)
                text_features = text_encoder(input_ids, output_hidden_states=True)
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_text_features = text_features[0]
                text_features = text_features.hidden_states[-2]
                text_features_list.append(text_features)
            text_features = torch.concat(text_features_list, dim=-1)
        else:
            assert False
        if to_numpy:
            text_features = text_features.contiguous().cpu().numpy()
            pooled_text_features = pooled_text_features.contiguous().cpu().numpy()
        return text_features, pooled_text_features
