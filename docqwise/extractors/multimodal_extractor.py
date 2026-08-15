"""Multimodal extraction — uses vision LLMs (Qwen2-VL, GPT-4o) for image-based documents.

For scanned documents, complex layouts, charts, handwriting — where text extraction fails.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Optional

from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import FieldResult, ExtractionResult

logger = logging.getLogger("docqwise")


class MultimodalExtractor:
    """Extract fields from document images using vision LLMs.

    Local models (via Ollama — free, offline):
        qwen2-vl, gemma3, llava, llava-llama3, bakllava,
        moondream, minicpm-v, cogvlm2, internvl2

    API models (cloud):
        gpt-4o, gpt-4o-mini, gpt-4-turbo (OpenAI)
        gemini-pro-vision, gemini-1.5-flash (Google — OpenAI-compatible)
        qwen-vl-plus, qwen-vl-max (Alibaba — OpenAI-compatible)

    Any OpenAI-compatible vision endpoint works.

    Usage:
        # Ollama local
        ext = MultimodalExtractor(model="qwen2-vl")
        ext = MultimodalExtractor(model="gemma3")
        ext = MultimodalExtractor(model="llava")

        # OpenAI API
        ext = MultimodalExtractor(model="gpt-4o", api_key="sk-...")

        # Any API
        ext = MultimodalExtractor(model="model", base_url="http://localhost:8000/v1")
    """

    # Models known to run on Ollama locally
    OLLAMA_VISION_MODELS = {
        "qwen2-vl", "qwen2.5-vl", "gemma3", "llava", "llava-llama3",
        "bakllava", "moondream", "minicpm-v", "cogvlm2", "internvl2",
        "llava-phi3", "nanollava",
    }

    # Models known to be API-based
    API_VISION_MODELS = {
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview",
        "gemini-pro-vision", "gemini-1.5-flash", "gemini-1.5-pro",
        "qwen-vl-plus", "qwen-vl-max", "claude-3-opus", "claude-3-sonnet",
    }

    # Models that can be loaded via HuggingFace transformers locally
    HF_VISION_MODELS = {
        "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct",
        "google/gemma-3-4b-it", "google/gemma-3-12b-it",
        "OpenGVLab/InternVL2-8B", "OpenGVLab/InternVL2-4B",
        "openbmb/MiniCPM-V-2_6", "vikhyatk/moondream2",
        "microsoft/Florence-2-large", "microsoft/Florence-2-base",
        "llava-hf/llava-v1.6-mistral-7b-hf",
    }

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 base_url: str = None, quantize: str = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.quantize = quantize  # "4bit" or "8bit" for HF models
        self._hf_model = None
        self._hf_processor = None
        self._local_model = None

        # Detect if model is a local file path
        import os
        self._is_local_file = os.path.exists(model) if model else False

    def extract_from_image(self, image_path: str, schema: dict = None,
                            prompt: str = None) -> ExtractionResult:
        """Extract fields from a document image."""
        image_b64 = self._encode_image(image_path)

        if schema:
            extraction_prompt = self._build_schema_prompt(schema)
        elif prompt:
            extraction_prompt = prompt
        else:
            extraction_prompt = self._build_auto_prompt()

        try:
            response = self._call_vision_api(image_b64, extraction_prompt)
            parsed = self._parse_response(response)
            return self._build_result(parsed, image_path)
        except Exception as e:
            logger.error(f"Multimodal extraction failed: {e}")
            return ExtractionResult(
                doc_id="", source_path=image_path,
                fields={}, confidence=0.0,
                strategy_used="multimodal_failed",
                warnings=[str(e)],
            )

    def extract_from_pdf_page(self, pdf_path: str, page_num: int = 0,
                               schema: dict = None) -> ExtractionResult:
        """Render PDF page to image, then extract with vision LLM."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            image_bytes = pix.tobytes("png")
            doc.close()

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            if schema:
                prompt = self._build_schema_prompt(schema)
            else:
                prompt = self._build_auto_prompt()

            response = self._call_vision_api(image_b64, prompt)
            parsed = self._parse_response(response)
            return self._build_result(parsed, pdf_path)

        except Exception as e:
            logger.error(f"PDF multimodal extraction failed: {e}")
            return ExtractionResult(
                doc_id="", source_path=pdf_path,
                fields={}, confidence=0.0,
                strategy_used="multimodal_failed",
                warnings=[str(e)],
            )

    def describe_chart(self, image_path: str) -> dict:
        """Extract data and description from a chart/graph image."""
        image_b64 = self._encode_image(image_path)
        prompt = """Analyze this chart/graph image. Extract:
1. Chart type (bar, line, pie, scatter, etc.)
2. Title and axis labels
3. All data points as a structured table
4. Key insights or trends

Return as JSON:
{
    "chart_type": "...",
    "title": "...",
    "x_axis": "...",
    "y_axis": "...",
    "data": [{"label": "...", "value": ...}, ...],
    "insights": ["...", "..."]
}"""

        try:
            response = self._call_vision_api(image_b64, prompt)
            return self._parse_response(response)
        except Exception as e:
            return {"error": str(e)}

    def _call_vision_api(self, image_b64: str, prompt: str) -> str:
        """Route to correct vision backend based on model."""
        import os
        model_base = self.model.split(":")[0].lower()

        # Local model file (.gguf, .pt, .onnx, .tflite, or folder)
        if self._is_local_file:
            if self.model.endswith(".gguf"):
                return self._call_gguf_vision(image_b64, prompt)
            elif self.model.endswith((".pt", ".pth", ".bin", ".safetensors")):
                return self._call_local_pt_vision(image_b64, prompt)
            elif self.model.endswith(".onnx"):
                return self._call_onnx_vision(image_b64, prompt)
            elif self.model.endswith(".tflite"):
                return self._call_tflite_vision(image_b64, prompt)
            elif os.path.isdir(self.model):
                # Local folder with HF model files (config.json + model files)
                return self._call_hf_vision(image_b64, prompt)
            else:
                raise ValueError(f"Unsupported local model format: {self.model}")

        # Explicit Ollama URL
        if self.base_url and "11434" in self.base_url:
            return self._call_ollama_vision(image_b64, prompt)

        # HuggingFace model path (contains "/")
        if "/" in self.model and (self.model in self.HF_VISION_MODELS or
                                  (not self.api_key and not self.base_url)):
            return self._call_hf_vision(image_b64, prompt)

        # Known Ollama models
        if model_base in self.OLLAMA_VISION_MODELS:
            try:
                from urllib.request import urlopen
                urlopen("http://localhost:11434/api/tags", timeout=2)
                return self._call_ollama_vision(image_b64, prompt)
            except Exception:
                pass

        # API key or known API model
        if self.api_key or model_base in self.API_VISION_MODELS:
            return self._call_openai_vision(image_b64, prompt)

        # Custom base_url
        if self.base_url:
            return self._call_openai_vision(image_b64, prompt)

        # Default: Ollama → API
        try:
            from urllib.request import urlopen
            urlopen("http://localhost:11434/api/tags", timeout=2)
            return self._call_ollama_vision(image_b64, prompt)
        except Exception:
            return self._call_openai_vision(image_b64, prompt)

    def _call_openai_vision(self, image_b64: str, prompt: str) -> str:
        """Call OpenAI-compatible vision API."""
        from urllib.request import urlopen, Request
        import json as json_mod

        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }}
                ]
            }],
            "max_tokens": 2000,
        }

        req = Request(url, data=json_mod.dumps(payload).encode(), headers=headers)
        resp = urlopen(req, timeout=60)
        return json_mod.loads(resp.read())["choices"][0]["message"]["content"]

    def _call_ollama_vision(self, image_b64: str, prompt: str) -> str:
        """Call Ollama vision model (Qwen2-VL, LLaVA, Gemma3, etc.)."""
        from urllib.request import urlopen, Request
        import json as json_mod

        base = self.base_url or "http://localhost:11434"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        req = Request(f"{base}/api/generate",
                      data=json_mod.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=120)
        return json_mod.loads(resp.read()).get("response", "")

    def _call_hf_vision(self, image_b64: str, prompt: str) -> str:
        """Call HuggingFace vision model loaded locally on GPU."""
        if self._hf_model is None:
            self._load_hf_model()

        from PIL import Image
        import io

        # Decode base64 to PIL Image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process based on model type
        model_lower = self.model.lower()

        if "qwen" in model_lower and "vl" in model_lower:
            return self._hf_qwen_vl(image, prompt)
        elif "florence" in model_lower:
            return self._hf_florence(image, prompt)
        else:
            return self._hf_generic(image, prompt)

    def _load_hf_model(self):
        """Load HuggingFace vision model to GPU."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}

        if self.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantize == "8bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["device_map"] = "auto"

        self._hf_processor = AutoProcessor.from_pretrained(
            self.model, trust_remote_code=True,
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model, **kwargs,
        )

    def _hf_qwen_vl(self, image, prompt: str) -> str:
        """Qwen2-VL / Qwen2.5-VL inference."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self._hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._hf_processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            output_ids = self._hf_model.generate(**inputs, max_new_tokens=2000)
        output = self._hf_processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:],
                                                  skip_special_tokens=True)[0]
        return output

    def _hf_florence(self, image, prompt: str) -> str:
        """Microsoft Florence-2 inference."""
        inputs = self._hf_processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            output_ids = self._hf_model.generate(**inputs, max_new_tokens=2000)
        return self._hf_processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def _hf_generic(self, image, prompt: str) -> str:
        """Generic HuggingFace vision model inference."""
        inputs = self._hf_processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            output_ids = self._hf_model.generate(**inputs, max_new_tokens=2000)
        return self._hf_processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def unload(self):
        """Free GPU memory."""
        self._hf_model = None
        self._hf_processor = None
        self._local_model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ══════════════════════════════════════
    # LOCAL MODEL FILE BACKENDS
    # ══════════════════════════════════════

    def _call_gguf_vision(self, image_b64: str, prompt: str) -> str:
        """Load .gguf model file via llama-cpp-python.

        Install: pip install llama-cpp-python
        Models: download from huggingface.co (search for GGUF vision models)
        """
        if self._local_model is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed.\n"
                    "  pip install llama-cpp-python\n"
                    "  GPU: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
                )
            self._local_model = Llama(
                model_path=self.model,
                n_ctx=4096,
                n_gpu_layers=-1,
            )

        # GGUF vision support (llava-style)
        from PIL import Image
        import io
        image_bytes = base64.b64decode(image_b64)

        output = self._local_model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }],
            max_tokens=2000,
        )
        return output["choices"][0]["message"]["content"]

    def _call_local_pt_vision(self, image_b64: str, prompt: str) -> str:
        """Load .pt/.pth/.safetensors model file via PyTorch.

        For custom-trained PyTorch vision models.
        Requires model architecture to be known.
        """
        import torch
        from PIL import Image
        import io

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self._local_model is None:
            # Try loading as a HuggingFace model with local files
            model_dir = str(self.model).rsplit("/", 1)[0] if "/" in str(self.model) else "."
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                self._hf_processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    model_dir, trust_remote_code=True,
                    torch_dtype=torch.float16, device_map="auto",
                )
                return self._hf_generic(image, prompt)
            except Exception:
                pass

            # Raw PyTorch model
            self._local_model = torch.load(self.model, map_location="cpu")

        return f"PyTorch model loaded from {self.model}. Use HuggingFace format for full inference."

    def _call_onnx_vision(self, image_b64: str, prompt: str) -> str:
        """Load .onnx model file via ONNX Runtime.

        Install: pip install onnxruntime (CPU) or onnxruntime-gpu (GPU)
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed.\n"
                "  CPU: pip install onnxruntime\n"
                "  GPU: pip install onnxruntime-gpu"
            )

        from PIL import Image
        import numpy as np
        import io

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self._local_model is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._local_model = ort.InferenceSession(self.model, providers=providers)

        # Get input shape
        input_info = self._local_model.get_inputs()[0]
        input_name = input_info.name
        shape = input_info.shape  # e.g. [1, 3, 224, 224]

        # Resize image
        h, w = shape[2] if len(shape) > 2 else 224, shape[3] if len(shape) > 3 else 224
        image = image.resize((w, h))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        outputs = self._local_model.run(None, {input_name: img_array})
        return str(outputs[0])

    def _call_tflite_vision(self, image_b64: str, prompt: str) -> str:
        """Load .tflite model file via TensorFlow Lite.

        Install: pip install tflite-runtime
        """
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                raise ImportError(
                    "TFLite not installed.\n"
                    "  pip install tflite-runtime\n"
                    "  OR: pip install tensorflow"
                )

        from PIL import Image
        import numpy as np
        import io

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self._local_model is None:
            self._local_model = tflite.Interpreter(model_path=self.model)
            self._local_model.allocate_tensors()

        input_details = self._local_model.get_input_details()
        output_details = self._local_model.get_output_details()

        # Resize to expected input
        h, w = input_details[0]["shape"][1], input_details[0]["shape"][2]
        image = image.resize((w, h))
        img_array = np.expand_dims(np.array(image).astype(np.float32) / 255.0, axis=0)

        self._local_model.set_tensor(input_details[0]["index"], img_array)
        self._local_model.invoke()
        output = self._local_model.get_tensor(output_details[0]["index"])
        return str(output)

    def _encode_image(self, path: str) -> str:
        """Read and base64-encode an image file."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_schema_prompt(self, schema: dict) -> str:
        fields_desc = json.dumps(schema, indent=2)
        return f"""Look at this document image carefully. Extract these fields:

{fields_desc}

For each field, find the exact value in the document.
For numbers, return numeric values without currency symbols.
For dates, return YYYY-MM-DD format.

Return ONLY valid JSON with field names as keys. No explanation."""

    def _build_auto_prompt(self) -> str:
        return """Look at this document image carefully.
1. Identify the document type (invoice, contract, receipt, form, report, etc.)
2. Extract ALL key-value pairs you can find
3. Extract any tables as structured data
4. Note any signatures, stamps, or handwritten content

Return ONLY valid JSON:
{
    "document_type": "...",
    "fields": {"field_name": "value", ...},
    "tables": [{"headers": [...], "rows": [[...], ...]}, ...],
    "has_signature": true/false,
    "has_handwriting": true/false
}"""

    def _parse_response(self, response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"raw_response": response}

    def _build_result(self, parsed: dict, source: str) -> ExtractionResult:
        fields = {}
        extracted = parsed.get("fields", parsed)
        if "document_type" in extracted:
            del extracted["document_type"]
        if "fields" in extracted:
            extracted = extracted["fields"]
        for name, value in extracted.items():
            if value is not None and value != "" and value != "null":
                fields[name] = FieldResult(
                    value=value,
                    confidence=0.93,
                    extraction_method="multimodal_vision",
                )
        confidence = sum(f.confidence for f in fields.values()) / max(len(fields), 1)
        return ExtractionResult(
            doc_id="",
            source_path=source,
            fields=fields,
            confidence=confidence,
            strategy_used="multimodal",
        )
