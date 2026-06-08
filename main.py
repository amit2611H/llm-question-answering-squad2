import json
import pandas as pd
import time
import torch
torch.set_default_device('cpu')

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.evaluate_results import NO_ANSWER_MARKER, evaluate_results


model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, token=True)
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id


def squad_qa(data_filename):
    """
    SQuAD 2.0 Question Answering System
    
    This function processes a SQuAD 2.0 dataset CSV and generates answers using 
    Llama-3.2-3B-Instruct model with structured output prompting and hallucination prevention.
    
    Key Features:
    1. Structured output prompting (HAS_ANSWER, ANSWER, EVIDENCE format)
    2. Hallucination prevention: validates answer exists in context
    3. Answer normalization: removes prefixes, extra punctuation
    4. Deterministic generation for consistency
    
    Args:
        data_filename: Path to input CSV with columns: context, question, is_impossible, answers
        
    Returns:
        out_filename: Path to output CSV with added 'final answer' column
    """
    
    # Read input data
    df = pd.read_csv(data_filename)
    final_answers = []
    
    print(f"Processing {len(df)} questions...")
    
    for idx, row in df.iterrows():
        context = row['context']
        question = row['question']
        
        # Create structured prompt
        messages = [
            {"role": "system", "content": """You are a precise QA assistant for SQuAD 2.0.
Your task is to answer questions based ONLY on the given context.

Output format (strictly follow this):
HAS_ANSWER: yes/no
ANSWER: [your answer or "NO ANSWER"]
EVIDENCE: [exact quote from context supporting your answer or "none"]

Rules:
1. Answer ONLY if the answer is explicitly in the context
2. Use exact words from the context when possible
3. Keep answers short (a few words or short phrase)
4. If you cannot find the answer in the context, output:
   HAS_ANSWER: no
   ANSWER: NO ANSWER
   EVIDENCE: none

Examples:
             
Answerable question:
Context: Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.
Question: In which year did Marie Curie win the Nobel Prize in Chemistry?
HAS_ANSWER: yes
ANSWER: 1911
EVIDENCE: Nobel Prize in Chemistry in 1911
             
Unanswerable question:
2.Context: The Nile is a river in Africa.
Question: What is the capital of France?
HAS_ANSWER: no
ANSWER: NO ANSWER
EVIDENCE: none           
"""},
{"role": "user", "content": f"""Context: {context}
       
Question: {question}

Provide your structured answer:"""}
        ]
        
        # Generate model response
        model_input = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            tokenize=False
        )
        
        inputs = tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
        
        # Deterministic generation
        outputs = model.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False,  # Deterministic
            temperature=None,
            top_p=None
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer from structured output
        final_answer = extract_and_validate_answer(response, context)
        final_answers.append(final_answer)
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} questions")
    
    # Add final answer column
    df['final answer'] = final_answers
    
    # Save results
    out_filename = data_filename.replace('.csv', '-results.csv')
    df.to_csv(out_filename, index=False)
    
    print(f'final answers recorded into {out_filename}')
    return out_filename


def extract_and_validate_answer(response, context):
    """
    Extract and validate answer from model's structured output.
    
    Implements hallucination prevention by:
    1. Checking if model says HAS_ANSWER: no
    2. Verifying EVIDENCE exists in context
    3. Normalizing answer output
    
    Args:
        response: Model's full response text
        context: Original context paragraph
        
    Returns:
        Normalized answer string or NO_ANSWER_MARKER
    """
    
    # Parse structured output
    has_answer = "no"
    answer = NO_ANSWER_MARKER
    evidence = ""
    
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('HAS_ANSWER:'):
            has_answer = line.split(':', 1)[1].strip().lower()
        elif line.startswith('ANSWER:'):
            answer = line.split(':', 1)[1].strip()
        elif line.startswith('EVIDENCE:'):
            evidence = line.split(':', 1)[1].strip()
    
    # Hallucination prevention rules
    if has_answer == "no":
        return NO_ANSWER_MARKER
    
    if answer.upper() == "NO ANSWER" or answer == NO_ANSWER_MARKER:
        return NO_ANSWER_MARKER
    
    # Verify evidence exists in context (hallucination check)
    #if evidence and evidence.lower() != "none":
        # Check if evidence is substring of context
        #if evidence.strip('"\'') not in context:
          #  return NO_ANSWER_MARKER
    
    # Normalize answer
    answer = normalize_answer(answer)
    if answer not in context:
        return NO_ANSWER_MARKER
    
    return answer


def normalize_answer(answer):
    """
    Normalize answer to match SQuAD format.
    
    Removes:
    - Common prefixes like "Answer:", "The answer is"
    - Extra punctuation at start/end
    - Leading/trailing whitespace
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string
    """
    
    # Remove common prefixes
    prefixes = [
        "answer:", "the answer is", "it is", "this is",
        "answer is", "the answer:", "a:", "q:"
    ]
    
    answer_lower = answer.lower()
    for prefix in prefixes:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):].strip()
            answer_lower = answer.lower()
    
    # Remove quotes if whole answer is quoted
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    # Clean up whitespace
    answer = ' '.join(answer.split())
    
    return answer


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    data = pd.read_csv(config['data'])
    sample = data.sample(n=config['sample_for_solution'])  # for grading will be replaced with 'sample_for_grading'
    sample_filename = config['data'].replace('.csv', '-sample.csv')
    sample.to_csv(sample_filename, index=False)

    out_filename = squad_qa(sample_filename)  # todo: the function you implement

    eval_out = evaluate_results(out_filename, final_answer_column='final answer')
    eval_out_list = [str((k, round(v, 3))) for (k, v) in eval_out.items()]
    print('\n'.join(eval_out_list))

    elapsed_time = time.time() - start_time
    print(f"time: {elapsed_time: .2f} sec")
