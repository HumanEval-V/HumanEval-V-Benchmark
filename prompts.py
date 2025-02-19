# Input: diagram, function signature. Output: code solution
PROMPT_V2C = '''You are an exceptionally intelligent coding assistant with a deep understanding of Python programming and a keen ability to interpret visual data. Your responses are consistently accurate, reliable, and thoughtful.

**Objective:**
You will be presented with a Python programming problem and an accompanying image. Please complete the function based on the provided image and code context.

**Note**
- Remember, the signature by itself does not contain the entire problem; the image provides critical details.
- Observe the image closely and determine how its visual elements correspond to the problem's inputs, outputs, operations, calculations, patterns (static/dynamic), and conditions.  
- Please generate the complete code solution, including its function signature and body, formatted in a single Python code block, **without any additional text or explanation**.

**Code Context:**
```python
{function_signature}
```
'''

# Input: diagram, function signature, cot instruction. Output: code solution
PROMPT_V2C_WITH_COT = '''You are an exceptionally intelligent coding assistant with a deep understanding of Python programming and a keen ability to interpret visual data. Your responses are consistently accurate, reliable, and thoughtful.

**Objective:**
You will be presented with a Python programming problem and an accompanying image. Please complete the function based on the provided image and code context.

**Note**
- Remember, the signature by itself does not contain the entire problem; the image provides critical details.
- Observe the image closely and determine how its visual elements correspond to the problem's inputs, outputs, operations, calculations, patterns (static/dynamic), and conditions.  
- First summarize the important clues or findings and write a step-by-step analysis. 
- Then generate the complete code solution, including the function signature and body, formatted in a single Python code block.

**Code Context:**
```python
{function_signature}
```
'''

# Input: problem specification, function signature. Output: code solution
PROMPT_T2C = '''**Instructions:**
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. Please complete the function based on the provided problem specification, code context, and accompanying image (if provided). Return the complete solution, including the function signature, in a single response, formatted within a Python code block.

**Problem Specification:**
```markdown
{problem_specification}
```

**Code Context:**
```python
{function_signature}
```
'''

# Input: diagram, function signature. Output: problem specification
PROMPT_V2T = '''**Instructions:**  
You will receive a Python programming problem and an accompanying image for analysis:

**Code Context:**
```python
{function_signature}
```

1. **Analyze the Function Signature**  
   Examine the provided function signature (its input, output, and goal) and identify any missing context. Remember, the signature by itself does not contain the entire problem; the image provides critical details.

2. **Examine the Image**  
   Observe the image closely and determine how its visual elements correspond to the problem's inputs, outputs, operations, calculations, patterns (static/dynamic), and conditions.  
   - First, describe the visual elements you see.  
   - Next, list the important facts from the image that are relevant for understanding the problem.  
   - Finally, deduce any missing information from the problem based on the image.

**Response Format:**  
Please structure your response in three main sections (use Markdown H1 headers):

1. **# Problem Restatement**  
   Provide a concise restatement of the problem, including relevant background and requirements.

2. **# Visual Facts**  
   List the facts directly observed from the image that are necessary for interpreting or solving the problem.

3. **# Visual Patterns**  
   Summarize any operations, calculations, static/dynamic patterns, and conditions inferred from these facts.

**Important Note:**  
- Clearly separate facts (what you directly see in the image) from patterns (what you infer based on those facts).  
- If complex visual information is difficult to express in plain language, use formal notation (mathematical or pseudo-code).  
- State only what you are sure of; do not introduce assumptions not supported by the image or give vague conclusions.  
- **Do not** include any code implementation in your response.
'''

# Input: function signature, prediction in previous iteration, execution feedback. Output: code solution
PROMPT_ITER_V2C = '''You are an exceptionally intelligent coding assistant with a deep understanding of Python programming and a keen ability to interpret visual data. Your responses are consistently accurate, reliable, and thoughtful.

**Objective:**
You will be presented with a Python programming problem, an accompanying image, and the problem analysis and code you previously generated. Your task is to refine both the **problem analysis** and the **code solution** based on execution feedback from the test cases.

**Code Context:**
```python
{function_signature}
```

**Previous Problem Analysis and Solution:**
```markdown
{previous_prediction}
```

**Execution Feedback:**
```
{execution_feedback}
```

**Note**
- Remember, the signature by itself does not contain the entire problem; the image provides critical details.
- Observe the image closely and determine how its visual elements correspond to the problem's inputs, outputs, operations, calculations, patterns (static/dynamic), and conditions.  
- Carefully review the execution feedback and analyze any errors or issues that arose during testing.
- Based on the feedback, refine your understanding of the problem and make necessary corrections. Ensure you revisit the previously neglected aspects from the image or problem analysis.
- !!You must NOT directly include the test cases from the feedback in your code. Doing so is considered cheating and invalidates the solution.!! Instead, improve the logic to handle all potential scenarios correctly.

**Your task is to generate:**
1. A revised version of the step-by-step problem analysis with an improved understanding of the visual details, operations, and conditions.
2. A refined Python code solution, formatted in a single code block, ensuring that it addresses the identified issues and passes all test cases **without hardcoding specific values from the feedback**.
'''

# Input: function signature, prediction in previous iteration, execution feedback. Output: problem specification
PROMPT_ITER_V2T = """You are an exceptionally intelligent coding assistant with a deep understanding of Python programming and a keen ability to interpret visual data. Your responses are consistently accurate, reliable, and thoughtful.

### Objective:
You will be presented with a Python programming problem, an accompanying image, and the **problem specification** you previously generated. Your task is to refine and generate a **new version of the problem specification** based on execution feedback from test cases.

### Code Context:
```python
{function_signature}
```

### Your Previous Version Problem Specification:
```markdown
{previous_prediction}
```

### Execution Feedback:
```
{execution_feedback}
```

### Instruction for Refining Problem Specification:
1. **Analyze the Function Signature**  
   - Examine the provided function signature (its input, output, and goal) and identify any missing context.
   - Remember, the signature by itself does not contain the entire problem; the image provides critical details.

2. **Examine the Image**
   Observe the image closely and determine how its visual elements correspond to the problem's inputs, outputs, operations, calculations, patterns (static/dynamic), and conditions.  
   - First, describe the visual elements you see.  
   - Next, list the important facts from the image that are relevant for understanding the problem.  
   - Finally, deduce any missing information from the problem based on the image.

3. **Execution Feedback Analysis**
   - Carefully review the execution feedback, especially error messages, unexpected outputs, or mismatches with the expected results. 
   - Analyze the issues that arose during testing, and consider how they may relate to aspects of the problem specification that were previously unclear, overlooked, or incorrectly defined.
   - !!You must NOT directly include the test cases from the feedback into the refined problem specification. Doing so is considered cheating and invalidates the refinement process.!!
   - Instead, generalize your understanding to address all possible cases comprehensively.

4. **Refine the Problem Specification**  
   Based on the execution feedback, revise your understanding of the problem.  
   - Clarify or update any ambiguous parts of the specification.  
   - Address missing or incorrect details in the initial problem specification that were revealed by the test cases.

### Response Format:
Please structure your response in three main sections (use Markdown H1 headers):

1. **# Problem Restatement**  
   Provide an updated, more precise version of the problem, reflecting the insights gained from the image and execution feedback.

2. **# Visual Facts**  
   List the visual elements from the image that are critical for understanding and solving the problem. Include any details that were previously neglected or misinterpreted.

3. **# Visual Patterns**  
   Summarize the operations, calculations, static/dynamic patterns, and conditions derived from the visual elements and execution feedback. Clearly explain how these patterns inform the solution.

### Important Note:
- Focus on refining the **problem specification**. **Do not** include any code implementation in your response.
- Ensure your refinement is based on problem understanding. **Do not** hardcode the test case values in your response.
"""
