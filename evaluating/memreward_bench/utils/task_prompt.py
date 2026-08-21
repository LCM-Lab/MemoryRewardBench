PROMPTS = dict(
    babilong =  dict(qa7 =  {
        'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'Daniel went to the bedroom. Daniel got the apple there. How many objects is Daniel carrying?\n'
            'Answer: one\n'
            '</example>\n'
            '<example>\n'
            'Mary grabbed the apple there. Mary gave the apple to John. How many objects is Mary carrying?\n'
            'Answer: none\n'
            '</example>\n'
            '<example>\n'
            'Sandra travelled to the hallway. Sandra picked up the milk there. Sandra took the apple there. '
            'Mary travelled to the garden. How many objects is Sandra carrying?\n'
            'Answer: two\n'
            '</example>\n',
        'post_prompt':
            'Your answer should contain only one word - $none$ or $number_of_objects$. '
            'Do not write anything else after that. Do not explain your answer.',
    },qa8 = {
      'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'Sandra travelled to the garden. Mary grabbed the milk there. What is Mary carrying?\n'
            'Answer: milk\n'
            '</example>\n'
            '<example>\n'
            'Mary travelled to the kitchen. Sandra travelled to the office. John travelled to the office. '
            'Sandra discarded the milk there. What is Sandra carrying?\n'
            'Answer: nothing\n'
            '</example>\n'
            '<example>\n'
            'Daniel grabbed the apple there. Mary went to the office. Daniel moved to the garden. '
            'Daniel grabbed the milk there. Mary went to the kitchen. What is Daniel carrying?\n'
            "Answer: apple,milk\n"
            "</example>\n",
        'post_prompt':
            'Your answer should contain only one or two words: $nothing$ or $object$ or $object_1$, $object_2$. '
            'Do not write anything else. Do not explain your answer.'
    }),
)