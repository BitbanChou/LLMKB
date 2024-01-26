from typing import List, Optional
from langchain.llms.base import LLM
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import torch


class LLMService(LLM):
    max_token: int = 1024
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "LLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        if "glm" in self.model.config._name_or_path:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=self.history,
                max_length=self.max_token,
                temperature=self.temperature,
            )
        elif "lama" in self.model.config._name_or_path:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            generation_output = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.max_token,
            )
            s = generation_output.sequences[0]
            # for ss in generation_output.sequences:
            #     print("reply："+ss)
            response = self.tokenizer.decode(s)
            start = response.find("Helpful Answer:") + len("Helpful Answer:")
            end = response.find("</s>", start)
            response = response[start:end].strip()
        elif "vicuna" in self.model.config._name_or_path:
            inputs = self.tokenizer([prompt])
            output_ids = self.model.generate(
                torch.as_tensor(inputs.input_ids).to(self.model.device),
                max_new_tokens=self.max_token,
            )
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print(response)

        return response

    def load_model(self, model_name_or_path: str = "THUDM/chatglm2-6b"):
        """
        load LLMs
        :return:
        """
        if "glm" in model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto").half()
            self.model = self.model.eval()
            # self.model = load_model_on_gpus(model_name_or_path,num_gpus=2)
        elif "lama" in model_name_or_path:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                          device_map="auto").half()
            self.model = self.model.eval()
            # self.model = load_model_on_gpus(model_name_or_path,num_gpus=2)
        elif "vicuna" in model_name_or_path:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                          device_map="auto").half()
            self.model = self.model.eval()
            # self.model = load_model_on_gpus(model_name_or_path,num_gpus=2)
