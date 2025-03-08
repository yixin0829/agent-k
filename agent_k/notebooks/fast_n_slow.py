# %% [markdown]
# # Fast & Slow Extraction
#
# This is a quick implementation of the Chain of Extraction + Fast & Slow thinking idea for text-to-JSON task.

# %% [markdown]
# ## Imports

# %%
import json
import operator
import sys
from ast import literal_eval
from enum import Enum
from typing import Annotated, Any, Literal

from IPython.display import Image, display
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from litellm import completion
from loguru import logger
from openai import OpenAI
from openai.types.beta import Assistant
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent_k.setup.load_43_101 import list_43_101_reports

# Configs
CLIENT = OpenAI()
TEMPERATURE = 0.5
MODEL = "gpt-4o-mini"
TOP_P = 0.95
LOG_LEVEL = "DEBUG"

# %%
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

# %% [markdown]
# ## Schemas


# %%
class DepositType(Enum):
    LATERITE_NICKEL = "Laterite nickel"
    UM_LAYERED_INTRUSION_NICKEL_COPPER_PGE = "U-M layered intrusion nickel-copper-PGE"
    KOMATIITE_NICKEL_COPPER_PGE = "Komatiite nickel-copper-PGE"
    UM_INTRUSION_NICKEL_COPPER_PGE = "U-M intrusion nickel-copper-PGE"
    SEDIMENT_HOSTED_COPPER_CO = "Sediment-hosted copper ± Co"
    MAFIC_ULTRAMAFIC_VMS = "Mafic-ultramafic VMS"
    FLUVIAL_PLACER_GOLD = "Fluvial placer gold"
    FLUVIAL_PLACER_PGE = "Fluvial placer PGE"
    BLACK_SHALE_URANIUM = "Black shale uranium"
    NOT_FOUND = "Not Found"

    @classmethod
    def from_string(cls, string_value):
        for member in cls:
            if member.value == string_value:
                return member
        return cls.NOT_FOUND


class DepositEnvironment(Enum):
    MAGMATIC = "Magmatic"
    SUPERGENE = "Supergene"
    VOLCANIC_BASIN_HYDROTHERMAL = "Volcanic basin hydrothermal"
    BASIN_HYDROTHERMAL = "Basin hydrothermal"
    BASIN_CHEMICAL = "Basin chemical"
    EROSIONAL = "Erosional"
    MAGMATIC_HYDROTHERMAL = "Magmatic hydrothermal"
    METAMORPHIC_HYDROTHERMAL = "Metamorphic hydrothermal"
    INFILTRATIONAL = "Infiltrational"
    BASIN_EVAPORATIVE = "Basin evaporative"
    NOT_FOUND = "Not Found"

    @classmethod
    def from_string(cls, string_value):
        for member in cls:
            if member.value == string_value:
                return member
        return cls.NOT_FOUND


class MineralSiteMetadata(BaseModel):
    mineral_site_name: str = Field(
        ..., description="The name of the mineral site that the report is about."
    )
    country: str = Field(
        default="Not Found",
        description="The country where the mineral site is located.",
    )
    state_or_province: str = Field(
        default="Not Found",
        description="The state or province where the mineral site is located.",
    )
    total_grade: str = Field(
        default="Not Found",
        description='The total grade of all nickel deposits in percentage format (e.g. 0.35% will be "0.35").',
    )
    total_tonnage: str = Field(
        default="Not Found",
        description='The total tonnage by summing all inferred, indicated, and measured nickel deposits in million tonnes (e.g. 123.4 Mt will be "123.4").',
    )
    top_1_deposit_type: DepositType = Field(
        default="Not Found",
        description="The most likely deposit type of the mineral site",
    )
    # Exclude this field as we can map it from the deposit type
    # top_1_deposit_environment: DepositEnvironment = Field(
    #     default="Not Found",
    #     description="The most likely deposit environment of the mineral site",
    # )


schema = MineralSiteMetadata.model_json_schema()
print(json.dumps(schema, indent=4))


# %%
class Example1(BaseModel):
    name: str
    address: str
    total_attendees: int
    oldest_attendee: str


class Example2(BaseModel):
    product_name: str
    product_type: str
    price: float
    discount: float


class Example3(BaseModel):
    address: str
    province: str
    country: str
    total_sales: float


schema_example1 = json.dumps(Example1.model_json_schema())
schema_example2 = json.dumps(Example2.model_json_schema())
schema_example3 = json.dumps(Example3.model_json_schema())
# replace "{" with "{{" and "}" with "}}" to be used in the multi-line prompt
schema_example1 = schema_example1.replace("{", "{{").replace("}", "}}")
schema_example2 = schema_example2.replace("{", "{{").replace("}", "}}")
schema_example3 = schema_example3.replace("{", "{{").replace("}", "}}")
print(schema_example1)
print(schema_example2)
print(schema_example3)

# %% [markdown]
# ## Prompts

# %%
SCHEMA_DECOMPOSE_SYS_PROMPT = """You are a helpful agent that groups entities in a JSON schema into two categories:
1. Simple entities in the JSON schema that can be extracted directly from the text.
2. Complex entities in the JSON schema that require reasoning or additional information to be extracted. Complex entities may include composite entities that need further decomposition or non-composite entities that require extra context for extraction.

You should enclose your reasoning within <thinking> XML tags and output the result within <output> XML tags."""

DECOMPOSE_USER_PROMPT_TEMPLATE = """# Example 1
Given the following JSON schema:
```
{{"properties": {{"name": {{"title": "Name", "type": "string"}}, "address": {{"title": "Address", "type": "string"}}, "total_attendees": {{"title": "Total Attendees", "type": "integer"}}, "oldest_attendee": {{"title": "Oldest Attendee", "type": "string"}}}}, "required": ["name", "address", "total_attendees", "oldest_attendee"], "title": "Example", "type": "object"}}
```
Output:
<thinking>
"name" and "address" are not complex entities and can be extracted directly from the text. "total_attendees" is likely a complex entity because it requires extracting individual attendees and counting them. Oldest attendee is a complex entity because it requires extracting the oldest attendee from the list of attendees.
</thinking>
<output>
1. Simple entities: ["name", "address"]
2. Complex entities: ["total_attendees", "oldest_attendee"]
</output>

# Example 2
Given the following JSOn schema:
```
{{"properties": {{"product_name": {{"title": "Product Name", "type": "string"}}, "product_type": {{"title": "Product Type", "type": "string"}}, "price": {{"title": "Price", "type": "number"}}, "discount": {{"title": "Discount", "type": "number"}}}}, "required": ["product_name", "product_type", "price", "discount"], "title": "Example2", "type": "object"}}
```
Output:
<thinking>
"product_name", "product_type", "price", and "discount" are all not complex entities and can be extracted directly from the text. "discount" is likely a complex entity because it requires extracting the discounted price and the original price from the text and then calculating the discount.
</thinking>
<output>
1. Simple entities: ["product_name", "product_type", "price"]
2. Complex entities: ["discount"]
</output>

# Example 3
Given the following JSON schema:
```
{{"properties": {{"address": {{"title": "Address", "type": "string"}}, "province": {{"title": "Province", "type": "string"}}, "country": {{"title": "Country", "type": "string"}}, "total_sales": {{"title": "Total Sales", "type": "number"}}}}, "required": ["address", "province", "country", "total_sales"], "title": "Example3", "type": "object"}}
```
Output:
<thinking>
"address", "province", and "country" are not complex entities and can be extracted directly from the text. "total_sales" is a complex entity because it requires extracting separate sale entities and then summing the sale values.
</thinking>
<output>
1. Simple entities: ["address", "province", "country"]
2. Complex entities: ["total_sales"]
</output>

# Example 4
Given the following JSON schema:
```
{json_schema}
```
Output:
"""


# Batch Extraction Assistant
PDF_AGENT_SYSTEM_PROMPT = """<SEE OPENAI ASSISTANT DASHBOARD>"""

PDF_AGENT_USER_PROMPT = """JSON schema provided: {json_schema}

Not take a deep breath and think step by step."""

# Deep Extraction Assistant
DEEP_EXTRACT_SYSTEM_PROMPT = """<SEE OPENAI ASSISTANT DASHBOARD>"""

DEEP_EXTRACT_USER_PROMPT = """Question: What's the {field} of the mineral site in the attached 43-101 report?
Description of {field}: {description}
Data type of {field}: {dtype}
Default value of {field}: {default}

Now take a deep breath and think step by step."""

# Optimizer Assistant
OPTIMIZER_SYSTEM_PROMPT = """<SEE OPENAI ASSISTANT DASHBOARD>"""

OPTIMIZER_USER_PROMPT = """Please correct the following extraction results based on the feedback and previous extraction messages.
Extraction results: {extraction_results}
Feedback: {feedback}
Previous extraction messages: {messages}
The provided JSON schema is: {json_schema}

Now take a deep breath and think step by step."""

VALIDATOR_SYSTEM_PROMPT = """You are a validation agent responsible for verifying extracted results against a given JSON schema and previous extraction messages. Your goal is to ensure data accuracy, correctness, and adherence to the expected structure.

### Guidelines:
1. Validate the extracted results against the JSON schema:
    - Ensure that all values conform to the expected data types.
    - Verify that extracted values match the correct format and structure.

2. Ensure the extracted values are precise:
    - Confirm that answers contain only the relevant data without any additional information, filler words, or extraneous details.

### Output Format:
- Reasoning: Enclose your thought process, validation steps, and identified issues in `<thinking>` XML tags.
- Feedback: Provide specific corrections, observations, or necessary adjustments in `<feedback>` XML tags.
- Final Validation Result:
    - If the extracted result is incorrect, output "NO" within `<output>` XML tags.
    - If the extracted result is correct, output "YES" within `<output>` XML tags.

Example Structure:
```
<thinking>
Detailed validation process, including schema checks and extracted value analysis.
</thinking>

<feedback>
Specific errors found or confirmation that the extraction is correct.
</feedback>

<output>
YES or NO
</output>
```

Ensure all responses are concise, structured, and directly aligned with the JSON schema validation criteria."""


VALIDATOR_USER_PROMPT = """Please validate the following extraction result based on the previous extraction messages.
Extraction result: {extraction_results}
Previous extraction messages: {messages}
The provided JSON schema is: {json_schema}

Now take a deep breath and think step by step."""

# %% [markdown]
# ## Helper Functions

# %% [markdown]
# ### Split JSON Schema


# %%
def split_json_schema(
    schema: dict, field_lists: list[list[str]], include_defs: bool = True
) -> list[dict]:
    """
    Splits a JSON schema into multiple schemas based on provided field lists,
    optionally including only the definitions referenced in each schema.

    Args:
        schema (dict): The input JSON schema.
        field_lists (list of list of str): Each inner list contains fields for a separate schema.
        include_defs (bool): If True, include referenced definitions from the original schema's $defs.

    Returns:
        list of dict: A list of JSON schemas corresponding to each field list.
    """
    # Extract properties, required fields, and definitions from the original schema
    original_properties = schema.get("properties", {})
    original_required = set(schema.get("required", []))
    original_defs = schema.get("$defs", {})

    schemas = []
    for field_list in field_lists:
        new_schema = {"type": "object", "properties": {}, "required": []}
        # Build the new schema based on the field list
        for field in field_list:
            if field in original_properties:
                new_schema["properties"][field] = original_properties[field]
            if field in original_required:
                new_schema["required"].append(field)

        # Remove 'required' key if it's empty
        if not new_schema["required"]:
            new_schema.pop("required")

        # Optionally include only the definitions referenced in the new schema
        if include_defs and original_defs:
            new_defs = {}
            for prop in new_schema["properties"].values():
                if "$ref" in prop:
                    # Expecting ref format: "#/$defs/DefinitionName"
                    ref_parts = prop["$ref"].split("/")
                    if len(ref_parts) >= 3 and ref_parts[1] == "$defs":
                        def_name = ref_parts[2]
                        if def_name in original_defs:
                            new_defs[def_name] = original_defs[def_name]
            if new_defs:
                new_schema["$defs"] = new_defs

        schemas.append(new_schema)

    return schemas


field_lists = [
    ["mineral_site_name", "state_or_province", "country"],
    ["total_grade", "total_tonnage"],
    ["top_1_deposit_type", "top_1_deposit_environment"],
    [],
]
result = split_json_schema(schema, field_lists)
for i, res in enumerate(result):
    print(f"Schema {i + 1}:", res)

# %% [markdown]
# ### Parse JSON Code Block


# %%
def parse_json_code_block(content: str) -> dict[str, Any]:
    """Parse the JSON code block from the assistant response."""
    try:
        json_code_block = content.split("<json>")[1].split("</json>")[0]
        return json.loads(json_code_block)
    except Exception as e:
        logger.error(f"Failed to parse JSON code block: {e}")
        logger.error(f"LLM response: {content}")
        return {}


# %% [markdown]
# ### Prompt OpenAI Assistant


# %%
def prompt_openai_assistant(assistant: Assistant, messages: list[dict]) -> str:
    thread = CLIENT.beta.threads.create(messages=messages)

    # Use the create and poll SDK helper to create a run and poll the status of
    # the run until it's in a terminal state.
    run = CLIENT.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    if run.status == "completed":
        messages = list(CLIENT.beta.threads.messages.list(thread_id=thread.id))
    try:
        message_content = messages[0].content[0].text
    except IndexError as e:
        logger.exception(f"{e}.`messages`: {messages}")

    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(
            annotation.text, f"[{index}]"
        )
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = CLIENT.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    logger.debug(message_content.value)
    # logger.debug("\n".join(citations))

    return message_content.value


# %% [markdown]
# ### Extract from PDF


# %%
def batch_extract(
    pdf_path: str,
    json_schema: dict,
) -> str:
    """
    Extract entities from a PDF file using OpenAI Assistant.
    """

    json_schema_str = json.dumps(json_schema)

    assistant = CLIENT.beta.assistants.retrieve("asst_dbMIxMYwSocPIKpZ3KLnadWB")

    filename_to_id_map = list_43_101_reports()
    filename = pdf_path.split("/")[-1]
    file_id = filename_to_id_map[filename]

    messages = [
        {
            "role": "user",
            "content": PDF_AGENT_USER_PROMPT.format(
                json_schema=json_schema_str,
            ),
            "attachments": [
                {
                    "file_id": file_id,
                    "tools": [
                        {"type": "file_search"},
                        {"type": "code_interpreter"},
                    ],
                }
            ],
        },
    ]

    content = prompt_openai_assistant(assistant, messages)

    return content


def deep_extract(pdf_path: str, field, default, description, dtype):
    """
    Extract ONE entity from a PDF file using OpenAI Assistant.
    """
    logger.info(f"Extracting {field} from {pdf_path}")
    # Use the same assistant for all deep extraction
    assistant = CLIENT.beta.assistants.retrieve("asst_50sbd2mNoNhaPecIKU34vXUP")
    # assistant = CLIENT.beta.assistants.create(
    #     name="MinMod Assistant (Deep Extraction - Map Reduce)",
    #     instructions=DEEP_EXTRACT_SYSTEM_PROMPT,
    #     tools=[{"type": "code_interpreter", "type": "file_search"}],
    #     model=MODEL,
    #     )

    # Get the OpenAI file ID
    filename_to_id_map = list_43_101_reports()
    filename = pdf_path.split("/")[-1]
    file_id = filename_to_id_map[filename]

    messages = [
        {
            "role": "user",
            "content": DEEP_EXTRACT_USER_PROMPT.format(
                field=field,
                description=description,
                dtype=dtype,
                default=default,
            ),
            "attachments": [
                {
                    "file_id": file_id,
                    "tools": [
                        {"type": "file_search"},
                        {"type": "code_interpreter"},
                    ],
                }
            ],
        },
    ]

    content = prompt_openai_assistant(assistant, messages)

    return content


# %% [markdown]
# # LangGraphs
# ## States, Nodes, and Routes

# %%


def viz_graph(graph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    pdf_path: str  # 43-101 report record ID
    json_schema: dict  # Predefined JSON schema (Assumed it's available)
    method: Literal["F&S", "DPE", "DPE + MAP_REDUCE"]

    # Populated by LLMs
    simple_entities: list[str]
    complex_entities: list[str]
    fast_schema: dict
    slow_schema: dict
    fast_extraction_agent_result: dict[str, Any]
    slow_extraction_agent_result_map_reduce: Annotated[list, operator.add]
    slow_extraction_agent_result: dict[str, Any]
    slow_extraction_validation: Literal["YES", "NO"]
    feedback: Annotated[list, add_messages]

    messages: Annotated[list, add_messages]
    final_extraction_result: MineralSiteMetadata


class ComplexEntityState(TypedDict):
    pdf_path: str
    entity_name: str
    description: str
    default_value: Any
    dtype: Literal["string", "number", "boolean", "array", "object"]


def schema_decompose(state: State):
    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": SCHEMA_DECOMPOSE_SYS_PROMPT},
            {
                "role": "user",
                "content": DECOMPOSE_USER_PROMPT_TEMPLATE.format(
                    json_schema=json.dumps(state["json_schema"])
                ),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    # Parse the <output> XML tags
    content = response.choices[0].message.content
    simple_entities = []
    complex_entities = []
    for line in content.split("\n"):
        if line.startswith("1. Simple entities:"):
            simple_entities = line.split(":")[1].strip()
            simple_entities = literal_eval(simple_entities)
        elif line.startswith("2. Complex entities:"):
            complex_entities = line.split(":")[1].strip()
            complex_entities = literal_eval(complex_entities)

    # TODO: Remove this once in production
    simple_entities = ["mineral_site_name", "country", "state_or_province"]
    complex_entities = ["total_grade", "total_tonnage", "top_1_deposit_type"]

    logger.debug(f"Response: {response.choices[0].message.content}")
    logger.debug(f"Simple entities: {simple_entities}")
    logger.debug(f"Complex entities: {complex_entities}")

    fast_schema, slow_schema = split_json_schema(
        state["json_schema"], [simple_entities, complex_entities]
    )

    return {
        "simple_entities": simple_entities,
        "complex_entities": complex_entities,
        "fast_schema": fast_schema,
        "slow_schema": slow_schema,
    }


def fast_and_slow_route(state: State):
    logger.info("Routing to the appropriate extraction agent")
    logger.debug(f"Fast schema: {state['fast_schema']}")
    logger.debug(f"Slow schema: {state['slow_schema']}")

    next_nodes = []

    if state["fast_schema"]["properties"]:
        next_nodes.append("fast_extraction_agent")
    if state["slow_schema"]["properties"]:
        if state["method"] == "DPE":
            next_nodes.append("slow_extraction_agent_dpe")
        elif state["method"] == "DPE + MAP_REDUCE":
            # Map reduce extraction of complex entities
            for entity_name, entity_schema in state["slow_schema"][
                "properties"
            ].items():
                next_nodes.append(
                    Send(
                        "slow_extraction_agent_map_reduce",
                        {
                            "pdf_path": state["pdf_path"],
                            "entity_name": entity_name,
                            "default_value": entity_schema.get("default", None),
                            "description": entity_schema.get("description", None),
                            "dtype": entity_schema.get("type", None),
                        },
                    )
                )
        else:
            # Default to batch extraction
            next_nodes.append("slow_extraction_agent")

    if not next_nodes:
        raise ValueError(
            "No next nodes found. Possibly because both simple and complex entities are empty."
        )

    return next_nodes


def fast_extraction_agent(state: State):
    logger.info("Batch extracting simple entities from the 43-101 report")

    fast_schema = state["fast_schema"]
    logger.debug(f"Fast schema: {fast_schema}")

    if fast_schema["properties"] == {}:
        raise ValueError(
            "No simple entities to extract. Returning empty dict as result."
        )

    content = batch_extract(state["pdf_path"], fast_schema)
    parsed_json = parse_json_code_block(content)

    return {"fast_extraction_agent_result": parsed_json}


def slow_extraction_agent(state: State):
    logger.info("Batch extracting complex entities from the 43-101 report")

    slow_schema = state["slow_schema"]
    logger.debug(f"Slow schema: {slow_schema}")

    if slow_schema["properties"] == {}:
        raise ValueError(
            "No complex entities to extract. Returning empty dict as result."
        )

    content = batch_extract(state["pdf_path"], slow_schema)
    parsed_json = parse_json_code_block(content)

    return {"slow_extraction_agent_result": parsed_json}


def slow_extraction_output_parser(
    content: str,
    entity_name: str,
    dtype: Literal["string", "number", "boolean", "array", "object"],
):
    # Corner cases
    if "not found" in content.lower():
        return "Not Found"

    try:
        parsed_output = content.split("<output>")[1].split("</output>")[0].strip()
    except IndexError:
        logger.exception(
            f"Error parsing <output> XML tags for {entity_name}\nContent: {content}"
        )

    if dtype == "number" and parsed_output != "Not Found":
        parsed_output = float(parsed_output)

    return parsed_output


def slow_extraction_agent_dpe(state: State):
    """Map reduce extraction of complex entities from the 43-101 report"""
    logger.info("Deep extraction of complex entities from the 43-101 report")

    slow_schema = state["slow_schema"]
    logger.debug(f"Slow schema: {slow_schema}")

    extraction_results = {}
    messages = []
    logger.info("Extracting entities from the 43-101 report")
    for entity_name, entity_schema in slow_schema["properties"].items():
        default_value = entity_schema.get("default", None)
        description = entity_schema.get("description", None)
        dtype = entity_schema.get("type", None)

        content = deep_extract(
            state["pdf_path"], entity_name, default_value, description, dtype
        )
        parsed_output = slow_extraction_output_parser(content, entity_name, dtype)

        extraction_results[entity_name] = parsed_output
        messages.append({"role": "assistant", "content": content})

    return {
        "slow_extraction_agent_result": extraction_results,
        "messages": messages,
    }


def slow_extraction_agent_map_reduce(state: ComplexEntityState):
    pdf_path = state["pdf_path"]
    entity_name = state["entity_name"]
    default_value = state["default_value"]
    description = state["description"]
    dtype = state["dtype"]

    content = deep_extract(pdf_path, entity_name, default_value, description, dtype)
    parsed_output = slow_extraction_output_parser(content, entity_name, dtype)

    return {
        "messages": [{"role": "assistant", "content": content}],
        "slow_extraction_agent_result_map_reduce": [{entity_name: parsed_output}],
    }


def slow_extraction_optimizer(state: State):
    slow_schema = state["slow_schema"]

    logger.info(
        "Correcting the existing slow extraction results based on the feedback and previous extraction messages"
    )
    assistant = CLIENT.beta.assistants.retrieve("asst_D2MTLBHWmh8sgYw2d90C4JyP")

    # Get the OpenAI file ID
    filename_to_id_map = list_43_101_reports()
    filename = state["pdf_path"].split("/")[-1]
    file_id = filename_to_id_map[filename]

    messages = [
        {
            "role": "user",
            "content": OPTIMIZER_USER_PROMPT.format(
                extraction_results=state["slow_extraction_agent_result"],
                feedback=state["feedback"],
                messages=state["messages"],
                json_schema=json.dumps(slow_schema),
            ),
            "attachments": [
                {
                    "file_id": file_id,
                    "tools": [
                        {"type": "file_search"},
                        {"type": "code_interpreter"},
                    ],
                }
            ],
        },
    ]

    content = prompt_openai_assistant(assistant, messages)
    parsed_json = parse_json_code_block(content)

    return {
        "slow_extraction_agent_result": parsed_json,
        "messages": [{"role": "assistant", "content": content}],
    }


def merge_map_reduce_results(state: State):
    merged_result = {}
    for d in state["slow_extraction_agent_result_map_reduce"]:
        for k, v in d.items():
            merged_result[k] = v

    return {"slow_extraction_agent_result": merged_result}


def validate_extraction_result(state: State):
    # Validate the extraction result
    logger.info("Validating slow extraction result")
    slow_schema = state["slow_schema"]

    # logger.debug(f"state messages: {state['messages']}")

    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VALIDATOR_USER_PROMPT.format(
                    extraction_results=state["slow_extraction_agent_result"],
                    messages=state["messages"],
                    json_schema=json.dumps(slow_schema),
                ),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    logger.debug(f"Response: {response.choices[0].message.content}")
    content = response.choices[0].message.content
    parsed_feedback = content.split("<feedback>")[1].split("</feedback>")[0]
    parsed_output = content.split("<output>")[1].split("</output>")[0]

    return {
        "slow_extraction_validation": parsed_output,
        "feedback": [{"role": "assistant", "content": parsed_feedback}],
    }


def validate_extraction_result_route(state: State):
    logger.info("Validating extraction result route")
    next_nodes = []
    if state["slow_extraction_validation"].lower().strip() == "no":
        next_nodes.append("slow_extraction_optimizer")
    elif state["slow_extraction_validation"].lower().strip() == "yes":
        next_nodes.append("slow_extraction_finish")
    else:
        # Go back to the slow extraction agent by default
        next_nodes.append("slow_extraction_optimizer")

    return next_nodes


def slow_extraction_finish(state: State):
    return {"slow_extraction_finish": True}


def extraction_synthesis(state: State):
    # synthesize the fast and slow extraction results into a single JSON object
    logger.info("Synthesizing extraction results")
    final_extraction_result = {
        **state["fast_extraction_agent_result"],
        **state["slow_extraction_agent_result"],
    }
    return {"final_extraction_result": MineralSiteMetadata(**final_extraction_result)}


# %% [markdown]
# ## Build the Graphs
# ### F&S Batch

# %%
# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("schema_decompose", schema_decompose)
graph_builder.add_node("fast_extraction_agent", fast_extraction_agent)
graph_builder.add_node("slow_extraction_agent", slow_extraction_agent)
graph_builder.add_node("extraction_synthesis", extraction_synthesis)
graph_builder.add_edge(START, "schema_decompose")
graph_builder.add_conditional_edges(
    "schema_decompose",
    fast_and_slow_route,
    {
        "fast_extraction_agent": "fast_extraction_agent",
        "slow_extraction_agent": "slow_extraction_agent",
    },
)
graph_builder.add_edge(
    ["fast_extraction_agent", "slow_extraction_agent"], "extraction_synthesis"
)
graph_builder.add_edge("extraction_synthesis", END)
# Compile the graph
graph = graph_builder.compile()

viz_graph(graph)

# %% [markdown]
# ### F&S with DPE (Deep Extraction)
# Validation loop.

# %%
# Build the graph
graph_builder_dpe = StateGraph(State)
graph_builder_dpe.add_node("schema_decompose", schema_decompose)
graph_builder_dpe.add_node("fast_extraction_agent", fast_extraction_agent)
graph_builder_dpe.add_node("slow_extraction_agent_dpe", slow_extraction_agent_dpe)
graph_builder_dpe.add_node("slow_extraction_optimizer", slow_extraction_optimizer)
graph_builder_dpe.add_node("validate_extraction_result", validate_extraction_result)
graph_builder_dpe.add_node("slow_extraction_finish", slow_extraction_finish)
graph_builder_dpe.add_node("extraction_synthesis", extraction_synthesis)
graph_builder_dpe.add_edge(START, "schema_decompose")
graph_builder_dpe.add_conditional_edges(
    "schema_decompose",
    fast_and_slow_route,
    {
        "fast_extraction_agent": "fast_extraction_agent",
        "slow_extraction_agent_dpe": "slow_extraction_agent_dpe",
    },
)
graph_builder_dpe.add_edge("slow_extraction_agent_dpe", "validate_extraction_result")
graph_builder_dpe.add_edge("slow_extraction_optimizer", "validate_extraction_result")
graph_builder_dpe.add_conditional_edges(
    "validate_extraction_result",
    validate_extraction_result_route,
    {
        "slow_extraction_optimizer": "slow_extraction_optimizer",
        "slow_extraction_finish": "slow_extraction_finish",
    },
)
graph_builder_dpe.add_edge(
    ["fast_extraction_agent", "slow_extraction_finish"], "extraction_synthesis"
)
graph_builder_dpe.add_edge("extraction_synthesis", END)
# Compile the graph
graph_dpe = graph_builder_dpe.compile()

viz_graph(graph_dpe)

# %% [markdown]
# ### F&S with DPE (Map Reduce)

# %%
# Build the graph
graph_builder_dpe_map_reduce = StateGraph(State)
graph_builder_dpe_map_reduce.add_node("schema_decompose", schema_decompose)
graph_builder_dpe_map_reduce.add_node("fast_extraction_agent", fast_extraction_agent)
graph_builder_dpe_map_reduce.add_node(
    "slow_extraction_agent_map_reduce", slow_extraction_agent_map_reduce
)
graph_builder_dpe_map_reduce.add_node(
    "slow_extraction_optimizer", slow_extraction_optimizer
)
graph_builder_dpe_map_reduce.add_node(
    "validate_extraction_result", validate_extraction_result
)
graph_builder_dpe_map_reduce.add_node("slow_extraction_finish", slow_extraction_finish)
graph_builder_dpe_map_reduce.add_node(
    "merge_map_reduce_results", merge_map_reduce_results
)
graph_builder_dpe_map_reduce.add_node("extraction_synthesis", extraction_synthesis)
graph_builder_dpe_map_reduce.add_edge(START, "schema_decompose")
graph_builder_dpe_map_reduce.add_conditional_edges(
    "schema_decompose",
    fast_and_slow_route,
    ["fast_extraction_agent", "slow_extraction_agent_map_reduce"],
)
graph_builder_dpe_map_reduce.add_edge(
    "slow_extraction_agent_map_reduce", "merge_map_reduce_results"
)
graph_builder_dpe_map_reduce.add_edge(
    "merge_map_reduce_results", "validate_extraction_result"
)
graph_builder_dpe_map_reduce.add_conditional_edges(
    "validate_extraction_result",
    validate_extraction_result_route,
    ["slow_extraction_optimizer", "slow_extraction_finish"],
)
graph_builder_dpe_map_reduce.add_edge(
    "slow_extraction_optimizer", "validate_extraction_result"
)
graph_builder_dpe_map_reduce.add_edge(
    ["fast_extraction_agent", "slow_extraction_finish"], "extraction_synthesis"
)
graph_builder_dpe_map_reduce.add_edge("extraction_synthesis", END)
# Compile the graph
graph_dpe_map_reduce = graph_builder_dpe_map_reduce.compile()

viz_graph(graph_dpe_map_reduce)

# %% [markdown]
# ## Run the Graphs
# Batch extraction for both simple and complex entities.


# %%
def extract_from_pdf(
    pdf_path: str,
    json_schema: dict,
    method: Literal["F&S", "DPE", "DPE + MAP_REDUCE"] = "DPE + MAP_REDUCE",
    recursion_limit: int = 12,
) -> MineralSiteMetadata:
    """
    Extract information from a PDF file using different extraction methods.

    Args:
        pdf_path (str): Path to the PDF file
        json_schema (dict): JSON schema defining the extraction fields
        method (str): Extraction method to use. One of "F&S", "DPE", or "DPE + MAP_REDUCE"
        recursion_limit (int): Maximum number of recursions for validation loop

    Returns:
        MineralSiteMetadata: Extracted information as a Pydantic model
    """
    try:
        if method == "F&S":
            result = graph.invoke(
                {
                    "pdf_path": pdf_path,
                    "json_schema": json_schema,
                    "method": method,
                }
            )
        elif method == "DPE":
            result = graph_dpe.invoke(
                {
                    "pdf_path": pdf_path,
                    "json_schema": json_schema,
                    "method": method,
                },
                {"recursion_limit": recursion_limit},
            )
        elif method == "DPE + MAP_REDUCE":
            result = graph_dpe_map_reduce.invoke(
                {
                    "pdf_path": pdf_path,
                    "json_schema": json_schema,
                    "method": method,
                },
                {"recursion_limit": recursion_limit},
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return result["final_extraction_result"]

    except GraphRecursionError as e:
        logger.error(f"Recursion Error: {e}")
        raise


if __name__ == "__main__":
    # Example usage of single file extraction
    result = extract_from_pdf(
        "data/raw/all_sources/43-101/021a794659a5c972b322e3bda39a1740793bb55223899eca13766ca7e84abdfded.pdf",
        schema,
        method="DPE + MAP_REDUCE",
    )
    print(result.model_dump_json(indent=4))


# %% [markdown]
# ### Vanilla F&S

# %%
if __name__ == "__main__":
    for s in graph.stream(
        {
            "pdf_path": "data/raw/all_sources/43-101/02c1e1ca772fe97dfc2d3a8923ccccd3a090b79c3c74a92e6f1ff093b08cbf0a57.pdf",
            "json_schema": schema,
            "method": "F&S",
        },
    ):
        print(s)

# %% [markdown]
# ### F&S with DPE

# %%
if __name__ == "__main__":
    try:
        result = graph_dpe.invoke(
            {
                "pdf_path": "data/raw/all_sources/43-101/02c1e1ca772fe97dfc2d3a8923ccccd3a090b79c3c74a92e6f1ff093b08cbf0a57.pdf",
                "json_schema": schema,
                "method": "DPE",
            },
            {"recursion_limit": 12},
        )
    except GraphRecursionError as e:
        logger.error(f"Recursion Error: {e}")

    print(result["final_extraction_result"].model_dump_json(indent=4))

# %% [markdown]
# ### F&S DPE with Map Reduce

# %%
if __name__ == "__main__":
    try:
        result = graph_dpe_map_reduce.invoke(
            {
                "pdf_path": "data/raw/all_sources/43-101/021a794659a5c972b322e3bda39a1740793bb55223899eca13766ca7e84abdfded.pdf",
                "json_schema": schema,
                "method": "DPE + MAP_REDUCE",
            },
            {"recursion_limit": 12},
        )
    except GraphRecursionError as e:
        logger.error(f"Recursion Error: {e}")

    print(result["final_extraction_result"].model_dump_json(indent=4))

# %% [markdown]
#
