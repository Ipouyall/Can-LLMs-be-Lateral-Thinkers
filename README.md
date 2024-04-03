# <p align="center">Can LLMs be Lateral Thinkers?</p>

<h2 align="center">
  <p><a href="https://semeval.github.io/SemEval2024">[SemEval@NAACL 2024]</a> Can LLMs be Lateral Thinkers?</p>
</h2>

<p align="center">
  <br>
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href="#"><img alt="Video" src="https://img.shields.io/badge/​-Video-red?logo=youtube&logoColor=FF0000"></a>
  <a href="#"><img alt="Slides" src="https://img.shields.io/badge/​-Slides-FFBB00?logo=googlesheets&logoColor=FFBB00"></a>
</p>

## Intro
This repo covers the implementation of the following paper:  **[uTeBC-NLP at SemEval-2024 Task 9: Can LLMs be Lateral Thinkers?]()** by [Pouya Sadeghi](https://www.linkedin.com/in/ipouyall), [Amirhossein Abaskohi](https://amirabaskohi.github.io/), and [Yadollah Yaghoobzadeh](https://yyaghoobzadeh.github.io/), accepted to SemEval@NAACL 2024.


## Abstract
Inspired by human cognition, [Jiang et al.(2023c)](https://aclanthology.org/2023.emnlp-main.885/) create a benchmark for assessing LLMs' lateral thinking—thinking outside the box. Building upon this benchmark, we investigate how different prompting methods enhance LLMs' performance on this task to reveal their inherent power for outside-the-box thinking ability. Through participating in SemEval-2024, task 9, Sentence Puzzle sub-task, we explore prompt engineering methods: chain of thoughts (CoT) and direct prompting, enhancing with informative descriptions, and employing contextualizing prompts using a retrieval augmented generation (RAG) pipeline. Our experiments involve three LLMs including GPT-3.5, GPT-4, and Zephyr-7B-beta. We generate a dataset of thinking paths between riddles and options using GPT-4, validated by humans for quality. Findings indicate that compressed informative prompts enhance performance. Dynamic in-context learning enhances model performance significantly. Furthermore, fine-tuning Zephyr on our dataset enhances performance across other commonsense datasets, underscoring the value of innovative thinking.

## Dataset

We use [BrainTeaser](https://github.com/1171-jpg/BrainTeaser) dataset in this paper. BrainTeaser is a multiple-choice question- answering task, designed to evaluate a model’s capability for lateral thinking and its ability to challenge default commonsense associations. Cre- ated to address the gap in the NLP community’s attention towards tasks requiring implicit and in- tricate reasoning, the dataset relies on human-like commonsense mechanisms. The authors devised a three-step approach to create the first lateral think- ing benchmark, involving data collection, distractor generation, and making adversarial examples. They produced 1,100 puzzles with detailed annotations. Assessing models’ lateral reasoning consistency, they enhanced BrainTeaser questions with seman- tic and contextual adjustments. Experiments with top-notch language models showed a significant performance difference from humans, particularly evident across adversarial formats, which aim to avoid cheating in scores by memorizing or previ- ously seen examples. The dataset includes 627 samples for sentence puzzles 5 and 492 samples for word puzzles 6. In the case of sentence puzzles utilized in our experiments, the average number of tokens in questions is 34.88, with an average of 9.11 tokens in the answers.

As mentioned in the paper, we created a modified version of this data which includes the path between the question and each option. This path between each option and question is called "thesis" and is generated by GPT-4 and revised by human evaluators. This dataset is available at [revised train kd](./experiments/SentencePuzzleKD/KD_train_gpt-4_revised.csv).

## Methodology

One approach to evaluating whether LLMs possess lateral thinking abilities is to prompt them explicitly for such capabilities. A key strategy involves providing hints about the task, signaling to the model that it should engage in unconventional thinking. In pursuit of this, we design three variations for task description: (I) Simple, which doesn't provide any special detail and serves as a base to provide evidence of how description could affect the model's performance, (II) Detailed, which would provide detailed information for the task and introduce common tricks to the LLM, and (III) Compressed, which is generated from the detailed variation and it just point out instead of detailed explanation.

We also consider CoT prompting as two main approaches:(I) Internal CoT and (II) External CoT. Internal CoT involves guiding the model through step-by-step thinking or incrementally posing questions to facilitate analytical consideration of each option. Our exploration of internal CoT encompasses two types: (I) Simple, and (II) Specified. In Simple Internal CoT, the model is prompted to think step-by-step without explicit specification of each intermediate step. Specified Internal CoT provides the model with explicitly outlined steps to follow in reaching its answer. Conversely, in External CoT, similar to specified-internal-CoT, we defined steps that the model should pass to reach the final answer, but instead of letting the model control the process, we prompt it to do one step in each inference and use the model's response to generate next prompts till we reach to the final answer. Our suggested intermediate reasoning steps, `find a path between the question and each answer option and then select the most logical one`, are independent for each question-option pair, and referred to as `thesis`. Then we would use them as context for each option of the riddle and prompt model to solve the riddle regarding provided contexts.

Furthermore we used in-context learning. In this approach, we let the model learn the task, using sample(s), known as few-shot prompting. In our few-shot experiments, we individually utilized three samples per question. We observed that employing static samples, as traditionally done in few-shot prompts, did not yield a significant performance boost, supporting few-shot results examined by Jiang et al. 2023. To overcome this limitation, we developed a RAG pipeline to select shots dynamically based on each question.

The following figure illustrates an overview of our approaches in solving the BrainTeaser riddles. In this setup, we have a direct prompt that asks the model to find the appropriate answer. To provide more information to the model, we can offer some task explanation, with the compressed version depicted in this figure. Finally, we utilize our RAG setup to provide the model with in-context examples. In some experiments, we also include the theses for each question-option pair in the prompt, serving as an unbiased link between the question and the option.

![image](https://github.com/Ipouyall/Can-LLMs-be-Lateral-Thinkers/assets/50926437/ca9a983b-b359-4dee-9707-6ab39c8fa948)

The next figure, illustrates RAG fusion. The four used variants include: (I) The original riddle, (II) Context reconstruction obtained from semantically reconstructed samples originating from the original riddle, (III) Context reconstruction derived from the original riddle, (IV) Context reconstructed from step 3, then we retrieve similar samples for each variant. In the end, we feed retrieved documents to a ranker to filter them based on similarity and usefulness.

![image](https://github.com/Ipouyall/Can-LLMs-be-Lateral-Thinkers/assets/50926437/ecc04331-9ab7-4ecb-8002-e0a4d6d486e7)

## Findings

The following table is our complete submission result for the post-evaluation phase on test split. In-context learning means using three shots dynamically selected by our RAG’s pipeline, in which: E) use Explanation, S) use Summarizer, R)use Ranker, and ord) using ordinary rag without explanation and ranker.

| Model                        | Thinking Method           | In-Context Learning | Task Description | Result |
|------------------------------|---------------------------|----------------------|------------------|--------|
| **GPT 3.5**                  | **Direct**               | -                    | None             | 72.5   |
| **GPT 3.5**                  |                           |                      | Compressed       | 72.5   |
| **GPT 3.5**                  |                           |                      | Detailed         | 75     |
| **GPT 3.5**                  | **Simple-Internal-CoT**  | -                    | None             | 70     |
| **GPT 3.5**                  |                           |                      | Compressed       | 70     |
| **GPT 3.5**                  |                           |                      | Detailed         | 72.5   |
| **GPT 3.5**                  | **Specified-Internal-CoT** | -                  | None             | 57.5   |
| **GPT 3.5**                  |                           |                      | Compressed       | 60     |
| **GPT 3.5**                  |                           |                      | Detailed         | 62.5   |
| **GPT 3.5**                  | **External-CoT**         | -                    | None             | 67.5   |
| **GPT 3.5**                  |                           |                      | Compressed       | 65     |
| **GPT 3.5**                  |                           |                      | Detailed         | 62.5   |
| **GPT 3.5**                  | **Simple-Internal-CoT**  | ES                   | Compressed       | 72.5   |
| **GPT 3.5**                  | **Direct**               | ES                   | Compressed       | 75     |
| **GPT 3.5**                  | **Direct**               | ES                   | Detailed         | 82.5   |
| **GPT 3.5**                  | **Simple-Internal-CoT**  | ER                   | Compressed       | 72.5   |
| **GPT 3.5**                  | **Direct**               | R                    | Compressed       | 82.5   |
| **GPT 3.5**                  | **Direct**               | R                    | Detailed         | 82.5   |
| **GPT 3.5**                  | **Direct**               | ord                  | None             | 85     |
| **GPT 3.5**                  | **Direct**               | ord                  | Compressed       | 85     |
| **GPT 3.5**                  | **Direct**               | ord                  | Detailed         | 85     |
| **GPT 3.5**                  | **Simple-Internal-CoT**  | ord                  | Compressed       | 77.5   |
| **GPT 3.5**                  | **Specified-Internal-CoT** | ord                | Compressed       | 67.5   |
| **GPT 4**                    | Direct                    | -                    | Detailed         | 95     |
| **GPT 4**                    | Simple-Internal-CoT       | -                    | Detailed         | 97.5   |
| **GPT 4**                    | Direct                    | ord                  | Compressed       | 92.5   |
| **Zephyr-7B-beta**           | **Direct**               | -                    | None             | 27.5   |
| **Zephyr-7B-beta**           |                           |                      | Detailed         | 32.5   |
| **Zephyr-7B-beta**           | **Simple-Internal-CoT**  | -                    | Compressed       | 37.5   |
| **Zephyr-7B-beta**           |                           |                      | Detailed         | 15     |
| **Zephyr-7B-beta**           |                           | ER                   | Compressed       | 40     |
| **Zephyr-7B-beta**           |                           | ES                   | Compressed       | 42.5   |
| **Zephyr-7B-beta**           |                           | ESR                  | Compressed       | 35     |
| **Zephyr-7B-beta**           |                           | ord                  | Compressed       | 25     |
| **Zephyr-7B-beta**           |                           | R                    | Compressed       | 22.5   |


## How to run?

In the `experiments` directory, list of all our experiments is created. Some expeirments include notebooks which you can follow the notebook for running it.

To run other python files, all of them are parameterized and for some of them a shell script exist. You can use the sheel script or run the following command to see the parameters of the script:

```
python3 FILE_NAME.py -h 
```
