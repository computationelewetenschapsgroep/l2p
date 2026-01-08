"""
This is a subclass (LLAMA) for abstract class (BaseLLM) that implements an interface
to interact with text generation models from LLAMA.

A YAML configuration file is required to specify model parameters and other
provider-specific settings. By default, the l2p library includes a configuration file
located at 'l2p/llm/utils/llm.yaml'.

Users can also define their own custom models and parameters by extending the YAML
configuration using the same format template.
"""


from typing_extensions import override
from .base import BaseLLM, load_yaml
from .utils.prompt_template import prompt_templates
from ..utils.pddl_parser import parse_heading as heading_parser
import warnings
import requests
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")


class LLAMA(BaseLLM):
    def __init__(
        self,
        model: str,
        model_path: str | None = None ,  # base directory of stored model
        config_path: str = "l2p/llm/utils/llm.yaml",
        provider: str = "llama",
        base_url: str = "http://localhost:11434/api/chat",
        api_key: str | None = None,  # only if model is affiliated w/ private repo
    ) -> None:

        self.url = base_url
        self.llm = model

    @staticmethod
    def enclosed_string(input:str, enclosure: str)->bool:
        stack = []
        for token in input.split():
            print(f"#### token: {token}") 
            if token == enclosure:
                stack.append(token)
        print(f"#### stack length: {len(stack)}")        
        return len(stack) %2 ==0
    
    @staticmethod
    def remove_open_enclosure(input:str, enclosure: str)-> str:
        if not __class__.enclosed_string(input, enclosure):
            input = input.replace(enclosure,"",1)
        return input
                      
    @staticmethod
    def post_process_response(response: dict) -> str:
        """Post-process the response from the LLM.

        Args:
            response (str): The raw response from the LLM.
        """
        content = response["message"]["content"] if "content" in response['message'] else str(response['message'])
        if "TYPES" in content:
            types_head = heading_parser(content, "TYPES")
            print(f"#################types_head: {types_head}")
        elif "Action Parameters" in content:
            types_head = heading_parser(content, "Action Parameters")
            print(f"#################action_params_head: {types_head}") 
        else:
            types_head = None    #processed_content = __class__.remove_open_enclosure(content, "```")
        
        if types_head:
            if types_head.count("```") != 2:
                prcoessed_types_head = types_head.replace("```","",1)
                content = content.replace(types_head,prcoessed_types_head)
                
        return content

    @override
    def query(
        self,
        prompt: str,
        system_prompt: str = None,
        end_when_error: bool = False,
        max_retry: int = 3,
        est_margin: int = 200,
    ) -> str:
       messages  = []
       messages.append({"role": "system","content":system_prompt})
       messages.append({"role":"user","content":prompt}) 
       print(f"messages: {messages}")
       r = requests.post(self.url, json={"model": self.llm, "messages": messages, "stream": False})
       
       result = r.json()
       print(f"llm response: {result}")
       return self.post_process_response(result)

    @override    
    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and system prompt and return the response.

        Args:
            system_prompt (str): The system prompt to send to the LLM
            prompt (str): The prompt to send to the LLM
        Returns:
            str: The response from the LLM
        """
        return self.query(system_prompt + "\n" + prompt)    
  
    @override
    def valid_models(self) -> list[str]:
        """Returns a list of valid model engines."""
        return []

    def reset_tokens(self) -> None:
        """Reset token counts."""
        self.in_tokens = 0
        self.out_tokens = 0
