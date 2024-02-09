compress_CoT_Enhanced = """\
You are given a brief example riddle and four options to choose the answer amongst them. \
A riddle is a question or statement intentionally phrased so as to require ingenuity in ascertaining its answer or meaning. \
Riddles are puzzles that need clever and logical thinking, and may try to trick you with \
default assumptions, social biases, and abnormally presenting the puzzle when there are always a logical solution.

Riddle: "{riddle}"

Options:
Option 1: "{option_1}"
Option 2: "{option_2}"
Option 3: "{option_3}"
Option 4: "None of the above options are correct"

To solve the riddle, think step by step for each option and consider providing an informative explanation or just the option number. \
Feel free to think creatively and consider alternative perspectives!\
At the end, announce the option you think is the best one in the format: 'Option 1' or 'Option 2' or 'Option 3' or 'Option 4':
"""

CoT_Enhanced = """\
You are given a riddle and four options to choose the answer amongst them. \
A riddle is a question or statement intentionally phrased so as to require ingenuity in ascertaining its answer or meaning. \
Different ideas can be used in riddles to trick you:
    1. Riddles often employ misdirection, leading you away from the actual solution.
    2. They include elements with double meanings, requiring a keen eye for words with dual interpretations.
    3. Metaphorical wordplay adds another layer, urging you to decipher figurative language.
    4. Look out for exaggeration, as riddles may present overly dramatic details to divert your attention.
    5. Common phrases and sayings may hide within the puzzle, demanding familiarity.
    6. Associations and irony play a crucial role, introducing unexpected connections.
    7. Numerical puzzles can also be part of the mystery, requiring you to decode their significance.
    8. Elemental imagery, drawn from nature, might hold key descriptors.
    9. Rhyming and sound clues can add a poetic dimension.
    10. Avoid sexism and sex cliché, for example, gender bias for jobs, based on their positions or their outcome.
    11. Riddle may try to present something impossible or in contradiction with the reality. Just consider alternative perspectives.
Although a clever solution is required, avoid supernatural solutions and keep your answer within the limits of realistic imagination. \
For example, having superhuman abilities or unusual events or things are mostly a not preferred choice unless that is a better solution. \
Now which of the following options is the answer to the following riddle:

Riddle: "{riddle}"

Options:
Option 1: "{option_1}"
Option 2: "{option_2}"
Option 3: "{option_3}"
Option 4: "None of the above options are correct"


Let's think step by step about each option and if it can be the answer of the riddle. \
At the end, announce the option you think is the best one in the format: 'Option 1' or 'Option 2' or 'Option 3' or 'Option 4':
"""
