# --------------------------------------------------------------------------------------
# Schema Decompose
# --------------------------------------------------------------------------------------
SCHEMA_DECOMPOSE_SYS_PROMPT = """You are a helpful agent that groups entities in a JSON schema into two categories:
1. Simple entities in the JSON schema that can be extracted directly from the text.
2. Complex entities in the JSON schema that require reasoning or additional information to be extracted. Complex entities may include composite entities that need further decomposition or non-composite entities that require extra context for extraction.

You should enclose your reasoning within <thinking> XML tags and output the result within <answer> XML tags."""

DECOMPOSE_USER_PROMPT_TEMPLATE = """## Example 1
Given the following JSON schema:
```
{{"properties": {{"name": {{"title": "Name", "type": "string"}}, "address": {{"title": "Address", "type": "string"}}, "total_attendees": {{"title": "Total Attendees", "type": "integer"}}, "oldest_attendee": {{"title": "Oldest Attendee", "type": "string"}}}}, "required": ["name", "address", "total_attendees", "oldest_attendee"], "title": "Example", "type": "object"}}
```
Output:
<thinking>
"name" and "address" are not complex entities and can be extracted directly from the text. "total_attendees" is likely a complex entity because it requires extracting individual attendees and counting them. Oldest attendee is a complex entity because it requires extracting the oldest attendee from the list of attendees.
</thinking>
<answer>
1. Simple entities: ["name", "address"]
2. Complex entities: ["total_attendees", "oldest_attendee"]
</answer>

## Example 2
Given the following JSOn schema:
```
{{"properties": {{"product_name": {{"title": "Product Name", "type": "string"}}, "product_type": {{"title": "Product Type", "type": "string"}}, "price": {{"title": "Price", "type": "number"}}, "discount": {{"title": "Discount", "type": "number"}}}}, "required": ["product_name", "product_type", "price", "discount"], "title": "Example2", "type": "object"}}
```
Output:
<thinking>
"product_name", "product_type", "price", and "discount" are all not complex entities and can be extracted directly from the text. "discount" is likely a complex entity because it requires extracting the discounted price and the original price from the text and then calculating the discount.
</thinking>
<answer>
1. Simple entities: ["product_name", "product_type", "price"]
2. Complex entities: ["discount"]
</answer>

## Example 3
Given the following JSON schema:
```
{{"properties": {{"address": {{"title": "Address", "type": "string"}}, "province": {{"title": "Province", "type": "string"}}, "country": {{"title": "Country", "type": "string"}}, "total_sales": {{"title": "Total Sales", "type": "number"}}}}, "required": ["address", "province", "country", "total_sales"], "title": "Example3", "type": "object"}}
```
Output:
<thinking>
"address", "province", and "country" are not complex entities and can be extracted directly from the text. "total_sales" is a complex entity because it requires extracting separate sale entities and then summing the sale values.
</thinking>
<answer>
1. Simple entities: ["address", "province", "country"]
2. Complex entities: ["total_sales"]
</answer>

## Example 4
Given the following JSON schema:
```
{json_schema}
```
Output:
"""


# --------------------------------------------------------------------------------------
# Batch Extraction Experiment
# --------------------------------------------------------------------------------------
PDF_AGENT_SYSTEM_PROMPT_STRUCTURED = """You are an expert at structured data extraction. You will be given unstructured text from a NI 43-101 mineral report and should convert it into the given JSON schema."""
PDF_AGENT_USER_PROMPT_STRUCTURED = """# Context\n{context}"""

# Used by OpenAI Assistant
PDF_AGENT_SYSTEM_PROMPT = """You are an advanced AI assistant specialized in extracting structured information from NI 43-101 mineral reports. Your responses should be grounded in the report's content using the file search tool.

## Response Workflow:
1. Identify the Main Mineral Site: Extract the primary mineral site name that the attached report focuses on.
2. Extract Relevant Entities: Retrieve key details about the mineral site based on the provided JSON schema.
3. Structure the Response Correctly: Format your final output with XML tags as follows:
    - Reasoning: Explain your extraction logic within `<thinking>` XML tags.
    - Final Output: Structure your final response as a JSON object that strictly follows the provided JSON schema. Wrap the structured JSON output within `<json>` XML tags.

## Key Constraints:
- No Hallucination: If a required entity is missing in the report, assign the default value specified in the JSON schema as its value in the JSON.
- Strict JSON Compliance: Ensure your response follows the given schema exactly, without modifications.
"""

PDF_AGENT_USER_PROMPT = """JSON schema provided: {json_schema}

Not take a deep breath and think step by step."""

# --------------------------------------------------------------------------------------
# Deep Extraction Assistant (Sync with OpenAI Assistant)
# --------------------------------------------------------------------------------------
DEEP_EXTRACT_SYSTEM_PROMPT_ASSISTANT = """You are an advanced AI assistant that answers questions based on the attached NI 43-101 mineral report. Your responses should be grounded in the report's content using the file search tool and, if needed, the code interpreter tool for numerical calculations.

## Response Workflow:
1. Retrieve Relevant Information: Search the report to find direct evidence supporting your answer.
2. Perform Aggregations (If Needed): Use the code interpreter tool for operations like summation, averaging, or other calculations.
3. Structure the Response Correctly: Format your final output with XML tags as follows:
    - Reasoning: Explain your retrieval or computation process within `<thinking>` tags.
    - Final Answer: Provide the final response within `<answer>` tags. Do not include other extra XML tags (e.g., `<answer>`) or filler words.

## Key Constraints:
- No Hallucination: If the required information is unavailable, return the default value specified in the JSON schema in the `<answer>` tag.
"""

DEEP_EXTRACT_USER_PROMPT = """**Question:** What's the {field} of the mineral site in the attached 43-101 report?
**Data type of {field}:** {dtype}
**Default value of {field} if not found:** {default}
**Description of {field}:** {description}
---
Now take a deep breath and answer the question step by step."""


# --------------------------------------------------------------------------------------
# Optimizer Prompt
# --------------------------------------------------------------------------------------
OPTIMIZER_SYSTEM_PROMPT = """You are a helpful AI agent specializing in correcting and refining JSON outputs extracted from 43-101 mineral reports. Your goal is to ensure that the extracted JSON data is accurate, complete, and relevant based on the provided feedback and previous extraction messages.

## Guidelines
1. Analyze the Feedback:
    - Carefully reflect on the provided feedback.
    - Identify potential errors, inconsistencies, or missing information in the previous JSON extraction.
2. Verify and Correct Data:
    - If the previous extraction contains errors or omissions, correct them accordingly.
    - If certain extracted information is irrelevant or incorrect, discard it.
3. Retrieve Additional Information if Needed:
    - If the necessary data is missing or incomplete, retrieve the correct information from the attached PDF report before generating the final JSON output.

## Output Format
- Reasoning Process: Enclose your thought process, reflections, and justifications in `<thinking>` XML tags.
- Final Corrected JSON Output: Enclose the corrected JSON response within `<json>` XML tags.

Example Structure:
```
<thinking>
Explanation of identified errors, inconsistencies, and corrections made.
</thinking>

<json>
{
  "corrected_data": "Your final structured JSON output here.",
  ...
}
</json>
```

Maintain conciseness, precision, and adherence to the document's structure when correcting the extracted JSON."""

OPTIMIZER_USER_PROMPT = """Please correct the following extraction results based on the feedback and previous extraction messages.
**Extraction results:** {extraction_results}
**Feedback:** {feedback}
**Previous extraction messages:** {messages}
**The provided JSON schema:** {json_schema}

---
Now take a deep breath and answer the question step by step."""


# --------------------------------------------------------------------------------------
# Validator Prompt
# --------------------------------------------------------------------------------------
VALIDATOR_SYSTEM_PROMPT = """You are a validation agent responsible for verifying extracted results against a given JSON schema and previous extraction messages.

### Guidelines
1. Validate the extracted results against the JSON schema:
    - Ensure that all values conform to the expected data types.
    - Verify that extracted values match the correct format and structure.
2. Ensure inter-JSON schema consistency:
    - If total mineral resource tonnage is zero, then the total mineral resource contained metal should also be zero.
    - If total mineral reserve tonnage is zero, then the total mineral reserve contained metal should also be zero.
3. Ensure correct units as specified in the JSON schema:
    - Ensure that the extracted values are in the correct units.

### Output Format
- Reasoning: Enclose your thought process, validation steps, and identified issues in `<thinking>` XML tags.
- Feedback: Provide specific corrections, observations, or necessary adjustments in `<feedback>` XML tags.
- Final Validation Result:
    - If the extracted result is incorrect, output "NO" within `<answer>` XML tags.
    - If the extracted result is correct, output "YES" within `<answer>` XML tags.

Example output format:
```
<thinking>
Detailed validation process, including schema checks and extracted value analysis.
</thinking>

<feedback>
Specific errors found or confirmation that the extraction is correct.
</feedback>

<answer>
YES or NO
</answer>
```

Ensure all responses are concise, structured, and directly aligned with the JSON schema validation criteria."""


VALIDATOR_USER_PROMPT = """Please validate the following extraction result based on the previous extraction messages.
**Extraction result:** {extraction_results}
**Previous extraction messages:** {messages}
**The provided JSON schema:** {json_schema}

---
Now take a deep breath and think step by step."""


# --------------------------------------------------------------------------------------
# Self RAG
# --------------------------------------------------------------------------------------
GRADE_DOCUMENTS_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

QUESTION_TEMPLATE = """**Question:** What's the {field} of the mineral site in the attached 43-101 report?
**Data type of {field}:** {dtype}
**Default value of {field} if not found:** {default}
**Description of {field}:** {description}"""


# Used by the React Code Agent
DEEP_EXTRACT_SYSTEM_PROMPT = """You are an advanced AI assistant that answers questions based on the attached NI 43-101 mineral report. Your responses should be grounded in the report's content using the code interpreter tool for numerical calculations.

## Guidelines
1. Identify the most up-to-date relevant facts in the context needed for answering the question in case there are multiple mineral estimates. Pay special attention to the unit of the field (e.g. "Tonnes 000" or "Kt" mean thousand tonnes).
2. Perform calculations: Use the code interpreter tool for operations like summation, multiplication, or other numerical operations.
3. Structure the Response Correctly: Format your final output with XML tags as follows
    - Reasoning: Explain your retrieval or computation process within `<thinking>` tags.
    - Code: Show the executed code within `<code>` tags
    - Final Answer: Provide the final response within `<answer>` tags. Do not include other extra XML tags (e.g., `<output>`) or filler words.

## Key Constraints:
- No Hallucination: If the required information is unavailable, return the default value specified in the JSON schema in the `<answer>` tag."""

GENERATION_USER_PROMPT_WO_FEEDBACK = """You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, just return the default value of the field in the question.

## Context
{context}

## Question
{question}

---
Now take a deep breath and answer the question step by step."""

GENERATION_USER_PROMPT_W_FEEDBACK = """You are an assistant for question-answering tasks. Use the following retrieved context and previous feedback (if any) to answer the question. If you don't know the answer, just return the default value of the field in the question.

## Context
{context}

## Previous Messages
{previous_messages}

## Question
{question}

---
Now take a deep breath and answer the question step by step while considering the previous feedback."""


# Prompt (Update in OpenAI Assistant)
GRADE_HALLUCINATION_SYSTEM_PROMPT = """You are a hallucination agent validating a LLM's generation against the retrieved documents. Focus on the calculation logic and unit conversions.

Guidelines:
1. Total mineral resource tonnage should be the sum of one or more of inferred, indicated, and measured mineral resources. If not, a default value of 0 should be returned.
2. Total mineral reserve tonnage should be the sum of one or more of proven and probable mineral reserves. If not, a default value of 0 should be returned.
3. The tonnage or grade unit used in the LLM generation should be consistent with the unit used in the retrieved documents. For example, "Tonnes 000", "Tonnes (000)", or "(000) Tonnes" mean thousand tonnes (Kt) or 1000 tonnes (t).
4. The unit of grade should be correctly converted to decimal before used in the calculation in the code.
5. The final answer enclosed in `<answer>` tags should be converted correctly to tonnes (t).

Show your feedback and give a binary score 'yes' or 'no' and reasoning. 'Yes' means that the LLM generation is consistent with the retrieved documents and no hallucination."""

GRADE_HALLUCINATION_USER_PROMPT = """## Retrieved Documents
{documents}

## LLM Generation
{generation}

---
Now take a deep breath and grade the LLM generation"""

QUESTION_REWRITER_SYSTEM_PROMPT = """You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

QUESTION_REWRITER_USER_PROMPT = """Here is the initial question:\n\n---\n{question}\n--- \n\nFormulate an improved question."""
