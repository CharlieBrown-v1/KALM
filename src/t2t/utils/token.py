current_state_token = ' #'
action_token = ' @'
next_state_token = ' ^'

current_state_id = 396
action_id = 732
next_state_id = 6228
end_id = 2
pad_id = 0
start_id = 1

eval_prompt = "Suppose you are playing a gaming of five balls with different colors under instruction. " \
    "Please explain the following content briefly.\n" \
    "Example:\n" \
    "Instruction: can you push the red ball to the left the blue ball. " \
    "Answer: move the red ball to the left of the blue ball.\n" \
    "Instruction: {instruction}. " \
    "Answer:" 

eval_prompt_inst = "Instruction: can you push the red ball to the left the blue ball. " \
    "Answer: move the red ball to the left of the blue ball.\n" \
    "Instruction: can you push the red ball to the left the blue ball. " \
    "Answer: move the red ball to the left of the blue ball.\n" \
    "Instruction: can you push the red ball to the left the blue ball. " \
    "Answer: move the red ball to the left of the blue ball.\n" \
    "Instruction: {instruction}. " \
    "Answer:" 

eval_prompt_traj = "Trajectory: {trajectory}." \
    "Answer: "

eval_prompt_t2t = "Suppose you are playing a gaming of five balls with different colors. " \
    "Please explain the following trajectory briefly.\n" \
    "Trajectory:{trajectory}\n" \
    "Answer:" 
