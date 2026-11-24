"""One local model, selected by name; share it across your simulated agents.

Usage (inside a Slurm GPU allocation):
    from model_pipeline import LocalLLM
    llm = LocalLLM("qwen8")  # or qwen4, deepseek8, moonlight16
    answer = llm.generate([{"role": "user", "content": "Hello"}])

The calling program owns each agent's conversation history. Nothing is served over HTTP.
"""
from pathlib import Path

# Names are intentionally short so teammates can change models easily.
MODELS = {
    "qwen4": "Qwen/Qwen3-4B",
    "qwen8": "Qwen/Qwen3-8B",                      # tested on Explorer L40S
    "deepseek8": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "moonlight16": "moonshotai/Moonlight-16B-A3B-Instruct",
}
ROOT = Path(__file__).resolve().parent


class LocalLLM:
    def __init__(self, model_name="qwen8"):
        if model_name not in MODELS:
            raise ValueError(f"Choose one of: {', '.join(MODELS)}")
        model_dir = ROOT / "models" / model_name
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Model not downloaded: {model_name}. Run MODEL={model_name} sbatch setup.sbatch"
            )

        # Import the large libraries only when actually loading a model.
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU. Start this from a Slurm GPU job: sbatch run.sbatch")

        self.torch = torch
        self.name = model_name
        custom_code = model_name == "moonlight16"
        print(f"Loading {MODELS[model_name]} from {model_dir}...", flush=True)
        # Moonlight's custom Python code is supplied by the model publisher.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=custom_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            trust_remote_code=custom_code,
        )
        self.model.eval()
        print(f"Ready on {torch.cuda.get_device_name(0)}", flush=True)

    def generate(self, messages, max_new_tokens=None):
        """Generate from a list of {role, content} messages; return generated text."""
        if not messages:
            raise ValueError("messages cannot be empty")
        messages = self._prepare_messages(messages)
        settings = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.name in ("qwen4", "qwen8"):
            settings["enable_thinking"] = False

        inputs = self.tokenizer.apply_chat_template(messages, **settings).to("cuda")
        # DeepSeek is a reasoning model, so allow more tokens by default.
        limit = max_new_tokens if max_new_tokens is not None else (
            400 if self.name == "deepseek8" else 150
        )
        generation = {"max_new_tokens": limit, "do_sample": False}
        if self.name == "deepseek8":
            generation.update(do_sample=True, temperature=0.6, top_p=0.95)
        if self.tokenizer.pad_token_id is not None:
            generation["pad_token_id"] = self.tokenizer.pad_token_id

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        new_ids = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _prepare_messages(self, messages):
        # DeepSeek-R1 authors recommend instructions in the user prompt,
        # rather than a separate 'system' message. Don't mutate caller history.
        copied = [dict(message) for message in messages]
        if self.name != "deepseek8":
            return copied
        instructions = "\n".join(
            message["content"] for message in copied if message["role"] == "system"
        )
        conversation = [m for m in copied if m["role"] != "system"]
        if instructions and conversation and conversation[0]["role"] == "user":
            conversation[0]["content"] = (
                f"Instructions: {instructions}\n\n{conversation[0]['content']}"
            )
        return conversation
