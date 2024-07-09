import json
from json import JSONDecodeError
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, TypedDict, TypeVar, Union, cast, overload
from operator import itemgetter

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    FunctionMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, Generation, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableMap
from langchain_core.tools import BaseTool

from langchain.agents.agent import AgentOutputParser
from langchain_experimental.llms.ollama_functions import (
    OllamaFunctions,
    convert_to_ollama_tool,
    _AllReturnType,
    parse_response,
    _is_pydantic_class,
    _is_pydantic_object,
    _BM,
    _DictOrPydanticClass,
    _DictOrPydantic,
)
from langchain_community.chat_models.ollama import ChatOllama


DEFAULT_SYSTEM_TEMPLATE = """You are a helpful assistant. You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema if any external tool is required for the query:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""  # noqa: E501

DEFAULT_RESPONSE_FUNCTION = {
    "name": "conversational_response",
    "description": (
        "Respond conversationally if no other tools should be called for a given query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Conversational response to the user.",
            },
        },
        "required": ["response"],
    },
}

class OllamaAIFunctionsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent action/finish.

    Is meant to be used with OpenAI models, as it relies on the specific
    function_call parameter from OpenAI to convey what tools to use.

    If a function_call parameter is passed, then that is used to get
    the tool and tool input.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "ollama-functions-agent"

    @staticmethod
    def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse an AI message."""
        
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        # Sample outputs for both non-tool and tool calls
        # AIMessage(content="Machine learning is a way that computers can learn from data without being explicitly programmed. It's like teaching a child to recognize pictures of cats and dogs by showing them many examples, so they can eventually identify new ones on their own.", 
        #          id='run-b5c3d86a-78ae-4e86-b506-f3d5db9403d9-0')
        # AIMessage(content='', 
        #   id='run-f4b2a91e-48fd-4967-883f-0939ab30919d-0', 
        #   tool_calls=[{'name': 'wikipedia', 'args': {'query': 'San Francisco weather'}, 'id': 'call_a95c73f442bd4ed3be0e04289e6d2e8d'}])
        # content='' id='run-5ebc6279-8cbb-4a35-9de5-61c6de43f016-0' tool_calls=[{'name': 'wikipedia', 'args': {'query': 'San Francisco weather'}, 'id': 'call_47178a1e90e949ddb5b69df091562780'}]
        
        # check if message has tool calls
        tool_calls = message.tool_calls
        if not message.tool_calls:
            # check if message content is not empty and JSON parseable
            try:
                # If it's JSON parseable, then it's a tool call
                message_content = json.loads(message.content)
                tool_calls = [message_content]
                
            except JSONDecodeError:
                # If it's not JSON parseable, then it's a regular response
                return AgentFinish(return_values={"output": message.content}, log=str(message))
        
        if not tool_calls:
            return AgentFinish(return_values={"output": message.content}, log=str(message))
        else:
            tool_call = tool_calls[0]
            print("Tool Call: ", tool_call)
            tool_name = tool_call["name"] if "name" in tool_call else tool_call["tool"]
            tool_input = tool_call["args"] if "args" in tool_call else tool_call["tool_input"]
            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{tool_name}` with `{tool_input}`\n{content_msg}\n"
            
            return AgentActionMessageLog(
                tool=tool_name,
                tool_input=tool_input,
                id = tool_call["id"] if "id" in tool_call else None,
                log=log,
                message_log=[message],
            )
    
    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        print("Result from Ollm before extracting: ", result)
        message = result[0].message
        print("Ollama func parse result: ", message)
        return self._parse_ai_message(message)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise ValueError("Can only parse messages")

class CustomOllamaFunctions(ChatOllama):
    """Function chat model that uses Ollama API."""

    tool_system_prompt_template: str = DEFAULT_SYSTEM_TEMPLATE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self.bind(functions=tools, **kwargs)

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]:
        ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Pydantic schema (include_raw=False):
            .. code-block:: python

                from langchain_experimental.llms import OllamaFunctions
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = OllamaFunctions(model="phi3", format="json", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Pydantic schema (include_raw=True):
            .. code-block:: python

                from langchain_experimental.llms import OllamaFunctions
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = OllamaFunctions(model="phi3", format="json", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: dict schema (method="include_raw=False):
            .. code-block:: python

                from langchain_experimental.llms import OllamaFunctions, convert_to_ollama_tool
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_ollama_tool(AnswerWithJustification)
                llm = OllamaFunctions(model="phi3", format="json", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }


        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        llm = self.bind_tools(tools=[schema], format="json")
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticOutputParser(
                pydantic_object=schema
            )
        else:
            output_parser = JsonOutputParser()

        parser_chain = RunnableLambda(parse_response) | output_parser
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser_chain, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | parser_chain

    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        ollama_messages: List = []
        for message in messages:
            role = ""
            content = ""
            if isinstance(message, HumanMessage) or isinstance(message, ToolMessage):
                role = "user"
                if isinstance(message, ToolMessage):
                    tool_name = message.name
                    content = f"Analyse this response taken from External tool {tool_name} to answer my query: "
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            images = []
            if isinstance(message.content, str):
                content += message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "image_url":
                        if isinstance(content_part.get("image_url"), str):
                            image_url_components = content_part["image_url"].split(",")
                            # Support data:image/jpeg;base64,<image> format
                            # and base64 strings
                            if len(image_url_components) > 1:
                                images.append(image_url_components[1])
                            else:
                                images.append(image_url_components[0])
                        else:
                            raise ValueError(
                                "Only string image_url content parts are supported."
                            )
                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )

            ollama_messages.append(
                {
                    "role": role,
                    "content": content,
                    "images": images,
                }
            )
        print(f"Ollama messages: {ollama_messages}")
        return ollama_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
            
        print("ENTERED CUSTOM OLLAMA FUNCTIONS GENERATE")
        print("kwargs:", kwargs)
        response_message = super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        print("EXECUTION DONE:", response_message)
        chat_generation_content = response_message.generations[0].text
        if not isinstance(chat_generation_content, str):
            raise ValueError("OllamaFunctions does not support non-string output.")
        try:
            parsed_chat_result = json.loads(chat_generation_content)
        except json.JSONDecodeError:
            raise ValueError(
                f"""'{self.model}' did not respond with valid JSON. 
                Please try again. 
                Response: {chat_generation_content}"""
            )
        print("Response: Ollama: ",parsed_chat_result)
        called_tool_name = (
            parsed_chat_result["tool"] if "tool" in parsed_chat_result else None
        )
        if (
            called_tool_name is None or called_tool_name == "conversational_response"
        ):
            if (
                "tool_input" in parsed_chat_result
                and "response" in parsed_chat_result["tool_input"]
            ):
                response = parsed_chat_result["tool_input"]["response"]
            elif "response" in parsed_chat_result:
                response = parsed_chat_result["response"]
            else:
                raise ValueError(
                    f"Failed to parse a response from {self.model} output: "
                    f"{chat_generation_content}"
                )
            
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=response,
                        )
                    )
                ]
            )

        called_tool_arguments = (
            parsed_chat_result["tool_input"]
            if "tool_input" in parsed_chat_result
            else {}
        )

        response_message_with_functions = AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name=called_tool_name,
                    args=called_tool_arguments if called_tool_arguments else {},
                    id=f"call_{str(uuid.uuid4()).replace('-', '')}",
                )
            ],
        )
        print("response message with func:", response_message_with_functions)
        return ChatResult(
            generations=[ChatGeneration(message=response_message_with_functions)]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        functions = kwargs.get("functions", [])
        if "functions" in kwargs:
            del kwargs["functions"]
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    "If `function_call` is specified, you must also pass a "
                    "matching function in `functions`."
                )
            del kwargs["function_call"]
        elif not functions:
            functions.append(DEFAULT_RESPONSE_FUNCTION)
        if _is_pydantic_class(functions[0]):
            functions = [convert_to_ollama_tool(fn) for fn in functions]
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )
        response_message = await super()._agenerate(
            [system_message] + messages, stop=stop, run_manager=run_manager, **kwargs
        )
        chat_generation_content = response_message.generations[0].text
        if not isinstance(chat_generation_content, str):
            raise ValueError("OllamaFunctions does not support non-string output.")
        try:
            parsed_chat_result = json.loads(chat_generation_content)
        except json.JSONDecodeError:
            raise ValueError(
                f"""'{self.model}' did not respond with valid JSON. 
                Please try again. 
                Response: {chat_generation_content}"""
            )
        called_tool_name = parsed_chat_result["tool"]
        called_tool_arguments = parsed_chat_result["tool_input"]
        called_tool = next(
            (fn for fn in functions if fn["name"] == called_tool_name), None
        )
        if called_tool is None:
            raise ValueError(
                f"Failed to parse a function call from {self.model} output: "
                f"{chat_generation_content}"
            )
        if called_tool["name"] == DEFAULT_RESPONSE_FUNCTION["name"]:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=called_tool_arguments["response"],
                        )
                    )
                ]
            )

        response_message_with_functions = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": called_tool_name,
                    "arguments": json.dumps(called_tool_arguments)
                    if called_tool_arguments
                    else "",
                },
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=response_message_with_functions)]
        )

    @property
    def _llm_type(self) -> str:
        return "ollama_functions"

""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
# _convert_agent_action_to_messages, _create_tool_message, format_to_ollama_tool_messages are Ollama Version which is similar to format_to_openai_function_messages
def _convert_agent_action_to_messages(
    agent_action: AgentAction, observation: str
) -> List[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return list(agent_action.message_log) + [
            _create_tool_message(agent_action, observation)
        ]
    else:
        return [AIMessage(content=agent_action.log)]

def _create_tool_message(
    agent_action: AgentAction, observation: str
) -> ToolMessage:
    """Convert agent action and observation into a tool message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        ToolMessage that corresponds to the original tool invocation
    """
    # To get tool call id - sample agent action
    # AgentActionMessageLog(tool='wikipedia', 
    #                       tool_input={'query': 'San Francisco weather'}, 
    #                       log="\nInvoking: `wikipedia` with `{'query': 'San Francisco weather'}`\n\n\n", 
    #                       message_log=[AIMessage(content='', 
    #                                              id='run-43a8d5d3-9afc-4aad-9aa1-2ec356280cfe-0', 
    #                                              tool_calls=[{'name': 'wikipedia', 'args': {'query': 'San Francisco weather'}, 'id': 'call_b9d0e49b6e44499181e47c5c7fbf1406'}])])
    
    tool_calls = agent_action.message_log[-1]
    tool_call_id = tool_calls.tool_calls[0]["id"] if tool_calls.tool_calls else str(uuid.uuid4())
    
    print("Agent Action in create tool message func: ", agent_action)
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    
    return ToolMessage(
        name=agent_action.tool,
        content=content,
        tool_call_id=tool_call_id
    )

def format_to_ollama_tool_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]], last_n_messages: int = None
) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations
        last_n_messages: Number of messages to include in the output. If None, include all previous intermediate steps.

    Returns:
        list of messages to send to the LLM for the next prediction

    Additional Information:
        last_n_messages is used to limit the number of messages that are sent to the LLM. This is useful when the LLM has small
        context windows like 8k context and small <=8B param models (Because too much information causes LLM to get confused). 
    """
    messages = []
    
    if last_n_messages is not None:
        intermediate_steps = intermediate_steps[-last_n_messages:]

    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages

""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
# Creating Ollama Functions Agent similar to openai functions agent
def fake_function(x: str) -> str:
    print(f"Fake function| Prompt: {x}")
    return x
    
def create_ollama_functions_agent(
    llm: OllamaFunctions, prompt: ChatPromptTemplate, last_n_messages: int = None
) -> Runnable:
    
    """
        Create an agent that uses Ollama Based LLMs. Similar to OpenAI Functions Agent.
        
        args:
            llm: OllamaFunctions object
            prompt: ChatPromptTemplate object
            last_n_messages: Number of messages to include in the output. If None, include all previous intermediate steps.
        
        returns:
            agent: Runnable object
    """
    
    # Check that the prompt has the necessary input variables
    if "agent_scratchpad" not in (
        prompt.input_variables + list(prompt.partial_variables)
    ):
        raise ValueError(
            "Prompt must have input variable `agent_scratchpad`, but wasn't found. "
            f"Found {prompt.input_variables} instead."
        )
    
    # print the dictionary entering the agent
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_ollama_tool_messages(
                x["intermediate_steps"], last_n_messages=last_n_messages
            )
        )
        |
        RunnablePassthrough.assign(
            print_input=lambda x: print(f"Agent Input - Inside agent chain: {x}") # Debugging
        )
        | prompt
        | # print prompt
        RunnableLambda(fake_function) # Debugging
        | llm
        | OllamaAIFunctionsAgentOutputParser()
    )
    return agent

""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
# Create the chat prompt template for ollama
def get_ollama_custom_chatPrompt():
    """
    Creates a Chat Prompt suitable for Custom Ollama Functions.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return prompt

""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
# Return system message for ollama with the tools
def get_ollama_tools_system_message(tools: List[BaseTool]):
    """
    Create a system message for the Ollama functions agent.
    
    Args:
    tools: List of tools to include in the system message.
    
    """

    # Convert function definitions to Dicts (tool name, description, parameters)
    functions = [convert_to_ollama_tool(fn) for fn in tools]
    functions.append(DEFAULT_RESPONSE_FUNCTION)
    print("functions:", functions)

    # Creating System Message Prompt Template for the tools
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(DEFAULT_SYSTEM_TEMPLATE)
    system_message = system_message_prompt_template.format(
        tools=json.dumps(functions, indent=2)
    )
    print("sys mess:", system_message)
    return system_message

""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
