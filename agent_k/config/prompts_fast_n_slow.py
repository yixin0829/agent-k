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

DEEP_EXTRACT_USER_PROMPT = """**Question:** What's the {field} of the mineral site in the attached 43-101 report?
**Data type of {field}:** {dtype}
**Default value of {field} if not found:** {default}
**Description of {field}:** {description}

Now take a deep breath and answer the question step by step."""

# Optimizer Assistant
OPTIMIZER_SYSTEM_PROMPT = """<SEE OPENAI ASSISTANT DASHBOARD>"""

OPTIMIZER_USER_PROMPT = """Please correct the following extraction results based on the feedback and previous extraction messages.
Extraction results: {extraction_results}
Feedback: {feedback}
Previous extraction messages: {messages}
The provided JSON schema is: {json_schema}

Now take a deep breath and answer the question step by step."""

VALIDATOR_SYSTEM_PROMPT = """You are a validation agent responsible for verifying extracted results against a given JSON schema and previous extraction messages. Your goal is to ensure data accuracy, correctness, and adherence to the expected structure.

### Guidelines:
1. Validate the extracted results against the JSON schema:
    - Ensure that all values conform to the expected data types.
    - Verify that extracted values match the correct format and structure.
2. Ensure the extracted values are precise:
    - Confirm that answers contain only the relevant data without any additional information, filler words, or extraneous details.
3. Ensure correct units as specified in the JSON schema:
    - Ensure that the extracted values are in the correct units.

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
