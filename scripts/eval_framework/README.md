# The Evaluation Framework

## Usage
Some actions are required prior to running the script to set the correct configuration
- Adjust `USE_MOCK_RESPONSE` variable
    - True: Use a mock LLM evaluator response (for testing purposes)
    - False: Use the API to get an evaluation from GPT 4.1-nano
- Adjust `SMALL_QUESTION_SET` variable: 
    - True: Use a small set of 5 questions and answers for evaluation. This can be helpful for testing the code out on the API without excessive amounts of tokens.
    - False: Use the full 100 question/answer set.
- Add applicable models to the `models` list in the `main` function: The code will repeat the evaluation for all models in the list, producing separate CSV files for later examination
    - **Make sure to set the `.name` attribute of your model (assuming based on `BaseF1Model`) to something unique!**
- In the current implementation, the working directory must be set to `/project_dir/scripts/eval_framework`

Once the configuration is set, the evaluation script can be run as follows:
```
python3 eval.py
```

## Evaluation Data
The data used in the evaluation set can be found in the `eval_set` subdirectory. Each line of the files in that subdirectory corresponds with the lines in the other files (e.g. the question on line 5 of questions.txt corresponds to the answer on line 5 of answers.txt)
- `questions.txt`: The full set of 100 questions
- `answers.txt`: The full set of 100 answers
- `notes.txt`: Notes for the evaluator LLM to better decipher how to interpret the responses for the 100 Q&A pairs (e.g. making sure the model being evaluated answers all parts of a question with multiple parts)
- `questions_small.txt`: Same as `questions.txt`, but only the first 5 questions.
- `answers_small.txt`: Same as `answers.txt`, but only the first 5 answers.
- `notes_small.txt`: Same as `notes.txt`, but only the first 5 notes.

## Scoring
The models are evaluated along 6 axes:
- **Relevance**: Is the provided response relevant to the current question?
- **Coherence**: Do the ideas presented in the response connect naturally?
- **Factuality**: Is the provided response factually correct? Could also be considered as "correctness".
- **Fluency**: Does the content of the response flow naturally (e.g. gramatically)?
- **Comprehensiveness**: Is there sufficient context provided by the model to fully understand the model's answer(s)?
- **Conciseness**: Is the model's response short and to the point, including only the necessary context?

*The scoring axes are defined in a bit further detail in the `SYSTEM_PROMPT` in `eval.py`.*

For a given model's response to a question, the Evaluator LLM will produce the following for each axis:
- **An integer rating of 0-100**: indicates the quality of the response for an axis, where 0 is a complete and utter failure, and 100 is what the evaluator perceives as perfection. 
- **A string-format justification**: allows for deeper understanding of the evaluator's reasoning behind the scores.

*Note: The axes mix both objective and subjective criteria. For instance, factuality (objective) axis may encounter more 100 values than comprehensiveness (subjective).*

## Output

A CSV file is produced for each model that is evaluated, with the name being `{model_name}_results.csv`, where `{model_name}` is the name assigned from the model's `name` attribute (defined as part of `BaseF1Model` class). The CSV file contains the corresponding question number and all of the scoring-related information mentioned above.