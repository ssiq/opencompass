from opencompass.datasets import (GSM8KDataset, generic_llmjudge_postprocess,
                                  gsm8k_dataset_postprocess,
                                  gsm8k_postprocess)
from opencompass.evaluator import (CascadeEvaluator, GenericLLMEvaluator,
                                   MATHVerifyEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

GSM8K_COT_EXAMPLES = """Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?
Let's think step by step
Answer: Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws. Their opponents score double the 2 pointers but half the 3 pointers and free throws. What's the total number of points scored by both teams added together?
Let's think step by step
Answer: Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201

Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?
Let's think step by step
Answer: When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?
Let's think step by step
Answer: For the first three baskets, the number of apples and oranges in one basket is 9+15=24
In total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.
Since there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.
The number of apples in the fourth basket is 9-2=7
There are also 15-2=13 oranges in the fourth basket
The combined number of oranges and apples in the fourth basket is 13+7=20
The fourth basket also contains 14-2=12 bananas.
In total, the fourth basket has 20+12=32 fruits.
The four baskets together have 32+114=146 fruits.
The answer is 146
"""

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {
                'role':
                'user',
                'content':
                GSM8K_COT_EXAMPLES +
                "\nQuestion: {question}\nLet's think step by step\nAnswer:"
            },
        ]),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

GRADER_TEMPLATE = """
    Please act as a grading expert. Judge whether the candidate answer is consistent with the standard answer.

    Evaluation criteria:
    1. The standard answer is correct. Do not solve the original problem again.
    2. The candidate answer may contain reasoning, code, execution output, or truncated residual text. Judge whether the answer expressed by the candidate is mathematically equivalent to the standard answer.
    3. Ignore formatting differences such as commas, currency symbols, units, or explanatory text when the numerical answer is the same.

    Grade the predicted answer as one of:
    A: CORRECT
    B: INCORRECT
    Just return the letter "A" or "B", with no text around it.

    <Original Question Begin>:
    {question}
    <Original Question End>

    <Gold Target Begin>:
    {reference}
    <Gold Target End>

    <Predicted Answer Begin>:
    {prediction}
    <Predicted End>

    Judging the correctness of the candidate answer:
""".strip()

cascade_evaluator = dict(
    type=CascadeEvaluator,
    rule_evaluator=dict(
        type=MATHVerifyEvaluator,
        pred_postprocessor=dict(type=gsm8k_postprocess),
    ),
    llm_evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {
                    'role':
                    'system',
                    'content':
                    "You are a helpful assistant who evaluates the correctness and quality of models' outputs."
                },
                {
                    'role': 'user',
                    'content': GRADER_TEMPLATE
                },
            ],
        ),
        dataset_cfg=dict(
            type=GSM8KDataset,
            path='opencompass/gsm8k',
            reader_cfg=gsm8k_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    parallel=False,
)

gsm8k_eval_cfg = dict(
    evaluator=cascade_evaluator,
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k-cascade-eval',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
        n=1,
    )
]
