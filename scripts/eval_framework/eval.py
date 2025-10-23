from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel
from csv import writer
import os

# IMPORT YOUR MODELS HERE!

SMALL_QUESTION_SET = True
USE_MOCK_RESPONSE = True
BASE_OUTPUT_PATH ="results.csv" # Don't mess with this: the CSV output file name is based upon model.name attribute.

# System Prompt: DO NOT TOUCH!
SYS_PROMPT = """
You are an AI assistant that specializes in evaluating responses from prototype conversational AI models focused on Formula 1 content across multiple axes on a numerical scale. You will focus solely on evaluating the prototype's response, and you will not create a new or refined answer to the questions.

# Evaluation Axes
relevance: Is the provided response relevant to the current question? For example, if a question is asked about the fastest lap at the 2023 Bahrain Grand prix, the prototype's response should not focus on the winner of the 2024 Spanish Grand Prix. Reward completeness of explanation without penalizing conciseness unless essential context is missing.
coherence: Do the ideas presented in the response connect naturally? For example, even if the prototype model correctly notes that a driver got disqualified from a race, it should not later say that the same driver won the race or received the most points.
factuality: Is the provided response factually correct? Do not penalize factual correctness for being unrelated, as that would be a relevance concern that is not scored on this axis.
fluency: Does the content of the response flow naturally? For example, assuming the question does not request different answer formatting, a response of "Carlos Sainz was the only driver who was not driving for Red Bull to win a Grand Prix in 2023." is preferable to "Carlos Sainz with Ferrari. 2023 winner.".
comprehensiveness: Is there sufficient context provided by the model to fully understand the model's answer(s)? For example, if the question asks about the purpose of DRS, it should ideally elaborate that DRS achieves higher top speed from an opened up rear wing that reduces drag on a car.
conciseness: Is the model's response short and to the point, including only the necessary context? For example, if the question just asks what DRS stands for in Formula 1, the model should provide the correct answer without going too much in depth regarding the meaning of DRS or its advantages. Note that some questions, especially those with multiple subquestions, will naturally require more wording and should not be penalized for doing so.

# User Prompt Components
Question: This was the question asked to the prototype model
Correct Answer: This is the ground truth answer that the prototype AI model should reference.
Notes: These are contextual notes to aid you, the evaluator LLM, in your evaluation along the axes, such as what qualifies as a partial answer, and what answers may be considered incorrect.
Response to Evaluate: This was the response provided by the prototype AI model that you should evaluate along the axes

# Expected Output
You, the LLM evaluator, will create a score and justification for each axis. 
Each axis is scored independently. A score of 100 represents excellent performance; 0 represents total failure to meet the criterion. 
Partial performance should be scored proportionally.
The key elements in the output are described below:
Score: Integer between 0 to 100 (inclusive) depicting the quality of the response along a certain axis. Higher is better.
Justification: String value depicting the justification behind the selected score for that particular axis. 1-3 sentences should be sufficient.


# Example
INPUT:
Question: What is the purpose of DRS in Formula 1?
Correct Answer: Reduce drag by opening up the rear wing in DRS zones, allowing for higher top speed and overtaking opportunities.
Notes for Evaluating: A correct answer can take many forms, though it should include at least some of the items mentioned in the ground truth answer.
Response to Evaluate: DRS allows drivers to achieve a higher top speed. They can use this to overtake rivals.

OUTPUT:
Factuality: 100 (The intended goal of DRS is in fact to achieve a higher top speed, and a higher top speed allows drivers to overtake rivals)
Coherence: 95 (The ideas generally flow naturally, with a higher top speed directly correlated to a better chance to overtake a rival.)
Fluency: 78 (The response could flow more naturally instead of being two distinct sentences, possibly connecting them together with a comma.)
Relevance: 100 (The response is very relevant to the question at hand, clearly showing an attempt at answering it. Even with the inclusion of using DRS to overtake rivals, it remains relevant in clarifying a common use of DRS)
Comprehensiveness: 73 (The response does touch upon the key ideas, but it could have elaborated on how DRS can only be used within DRS zones, for example.)
Conciseness: 98 (The answer is short and to the point. It could be condensed a bit further, but the current response is more than sufficient.)
""".strip()

# A basic class representing a model
class BaseF1Model:
    def __init__(self, name: str = "Eval_Basic"):
        self.name = name

    # OVERRIDE THIS FUNCTION IN YOUR MODELS!
    def predict(self, prompt: str) -> str:
        return f"A text response to the prompt: {prompt}"

# Subclass representing scoring info on each axis. DO NOT TOUCH!
class Eval_Score(BaseModel):
    score: int
    justification: str

# GPT 4.1 Nano Output Schema: DO NOT TOUCH!
class LLM_Evaluator_Output(BaseModel):
    relevance: Eval_Score
    coherence: Eval_Score
    factuality: Eval_Score
    fluency: Eval_Score
    comprehensiveness: Eval_Score
    conciseness: Eval_Score

# Loads the lines from applicable files (questions, answers, evaluator notes)
def load_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

# Create a "user" prompt for the evaluator LLM based on the question/answer/model response
def create_llm_user_prompt(question: str, answer: str, notes: str, model_answer: str) -> str:
    user_prompt = ""
    user_prompt += "Question: " + question + "\n"
    user_prompt += "Correct Answer: " + answer + "\n"
    user_prompt += "Notes for Evaluating: " + notes + "\n"
    user_prompt += "Response to Evaluate: " + model_answer + "\n"
    return user_prompt

# Create evaluations based upon a model's response to some question
def generate_response(client: OpenAI, question: str, answer: str, notes: str, model_answer: str) -> LLM_Evaluator_Output | dict:
    try:
        eval_response = client.responses.parse(
            model="gpt-4.1-nano",
            input=[
                {
                    "role": "system",
                    "content": SYS_PROMPT,
                },
                {
                    "role": "user",
                    "content": create_llm_user_prompt(question, answer, notes, model_answer),
                },
            ],
            text_format=LLM_Evaluator_Output
        )

        return eval_response
    except Exception as e:
        print("ERROR: Encountered an error while generating a response from OpenAI API: ", e)
        return None

# Local-only version of evaluation generation for testing purposes
# Note: this function has the same signature as generate_response, but does not use the OpenAI client.
def generate_response_mock(client: Optional[OpenAI], question: str, answer: str, notes: str, model_answer: str) -> LLM_Evaluator_Output | dict:
    return LLM_Evaluator_Output(
        relevance=Eval_Score(score=95, justification="Highly relevant."),
        coherence=Eval_Score(score=93, justification="Coherent and structured."),
        factuality=Eval_Score(score=82, justification="Mostly factual."),
        fluency=Eval_Score(score=96, justification="Fluent text."),
        comprehensiveness=Eval_Score(score=76, justification="Good, but could provide more context"),
        conciseness=Eval_Score(score=99, justification="Short and to the point.")
    )

# A wrapper function that repeatedly calls a response generation function for all question/answer pairs
def evaluate_all(client: OpenAI, questions: List[str], answers: List[str], notes: List[str], model: BaseF1Model, mock: bool = False) -> List[LLM_Evaluator_Output]:
    outputs = []
    eval_func = generate_response_mock if mock else generate_response
    try:
        from tqdm import tqdm
        iterator = range(len(questions))
        progress = tqdm(iterator, desc="INFO: Evaluating", unit="q")
        for i in progress:
            q, a, n = questions[i], answers[i], notes[i]
            model_answer = model.predict(q)
            output = eval_func(client, q, a, n, model_answer)
            if output == None:
                print("INFO: Halting execution...")
                return None
            outputs.append(output)
            progress.set_postfix({"idx": i+1})
    except ImportError:
        print("WARN: tdqm python module not available: logging will be disabled for this run")
        for q, a, n in zip(questions, answers, notes):
            model_answer = model.predict(q)
            output = eval_func(client, q, a, n, model_answer)
            if output == None:
                print("INFO: Halting execution...")
                return None
            outputs.append(output)
    return outputs

def main() -> None:
    models : List[BaseF1Model] = []
    # ADD YOUR MODELS INTO THE LIST HERE!!!
    # 2 BaseModels added here as an example
    #model1 = BaseF1Model() # Use default name: Eval_Basic
    #models.append(model1)

    model2 = BaseF1Model() # Use custom name: Testing
    model2.name = "Testing"
    models.append(model2)

    # END ADDING MODELS HERE

    # Print out the configuration we're using
    print(f"INFO: CONFIGURATION OPTIONS SET --- API Usage: {not USE_MOCK_RESPONSE} | Small Q/A Set: {SMALL_QUESTION_SET}")

    # Get the secret key for OpenAI API
    secret = os.environ.get("OPENAI_API_KEY")

    # If we're in mock mode, just warn the user: we don't need the secret for now
    if not secret and USE_MOCK_RESPONSE:
        print("WARN: Using mock response mode, but no OPENAI_API_KEY environment variable set. Make sure to add this for non-mock mode!")
    
    # If we're in normal mode and are missing a secret key, we can't perform the evaluation!
    elif not secret and not USE_MOCK_RESPONSE:
        print("ERROR: No OPENAI_API_KEY found in environment variables, and it's needed for standard evaluation.")
        exit()

    # Instantiate connection to OpenAI API
    client = OpenAI()

    # Load the questions/answers/notes
    base = Path("./eval_set")
    if SMALL_QUESTION_SET:
        questions = load_lines(base / "questions_small.txt")
        answers = load_lines(base / "answers_small.txt")
        notes = load_lines(base / "notes_small.txt")
    else: 
        questions = load_lines(base / "questions.txt")
        answers = load_lines(base / "answers.txt")
        notes = load_lines(base / "notes.txt")

    # Should be 100 of each item in the eval suite. If not, something's gone horribly wrong.
    if len(questions) != len(answers) or len(answers) != len(notes):
        print("ERROR: Expected the same number of questions/answers/notes!")
        exit()

    print(f"Loaded: {len(questions)} questions, {len(answers)} answers, {len(notes)} notes")

    # Testing LLM response functionality
    for model in models:
        outs = evaluate_all(client, questions, answers, notes, model, USE_MOCK_RESPONSE)

        # Write the evaluations to CSV files
        output_path = f"{model.name}_{BASE_OUTPUT_PATH}"
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            csvwriter = writer(f)
            csvwriter.writerow(["Question_Num","Relevance_Score","Coherence_Score","Factuality_Score","Fluency_Score",
                                "Comprehensiveness_Score","Conciseness_Score","Relevance_Reason","Coherence_Reason",
                                "Factuality_Reason","Fluency_Reason","Comprehensiveness_Reason","Conciseness_Reason"])
            for i, o in enumerate(outs, start=1):
                if not USE_MOCK_RESPONSE:
                    o = o.output_parsed
                csvwriter.writerow([i,o.relevance.score,o.coherence.score,o.factuality.score,o.fluency.score,o.comprehensiveness.score,o.conciseness.score,
                                    o.relevance.justification,o.coherence.justification,o.factuality.justification,o.fluency.justification,o.comprehensiveness.justification,o.conciseness.justification])
        
        print(f"INFO: Wrote evaluation results to {output_path}. Make sure to save this file! eval.py will overwrite this in the next run!")

if __name__ == "__main__":
    main()