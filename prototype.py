import gym
import numpy as np
import torch
import requests 
import json
import os
import time
from typing import Optional

import gym_pcgrl
from gym_pcgrl.envs.helper import get_string_map
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.probs.binary_prob import BinaryProblem
import network
from network import RewardNet



key = os.environ["CAMPUS_KI_API_KEY"]
url = "https://chat.kiconnect.nrw/api/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}"
}
MODEL = "gpt-6-studierende" 
REASONING_EFFORT = "none"
TIMEOUT_SECONDS = 300
OUT_FILE = "pairs.jsonl"

BINARY_PROBLEM = BinaryProblem()


def test_Connection():
    data = {
        #"model": "Mistral Small 3.2 24B Instruct 2506",
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Antworte bitte nur mit: Verbindung Erfolgreich!"
            },
        ],
    }
    if REASONING_EFFORT is not None:
        data["reasoning_effort"] = REASONING_EFFORT
    if REASONING_EFFORT == "none" or REASONING_EFFORT is None:
        data["temperature"] = 0.0

    response = requests.post(url, headers=headers, json=data, timeout=30)
    llm_result_text = response.json()["choices"][0]["message"]["content"]
    response.raise_for_status()
    print(llm_result_text)


def send_2_levels(level_A, level_B, game = None):
    """Takes two Levels (Level A and B) and a game name.  
    The levels are then evaluated by the LLM based on metrics specified for the given game.  
    Returns the raw response from the API call."""

    if game == "binary":
        # system_prompt = (
        # "You are an evaluator for 2D grid levels from the PCGRL binary benchmark. "
        # "Each level is a 14x14 grid. '#' is a wall, '.' is a walkable tile.\n\n"
        # "Definitions:\n"
        # "- Two walkable tiles are connected if they are direct horizontal or vertical neighbours (no diagonal moves).\n"
        # "- A region is a maximal group of connected walkable tiles.\n"
        # "- The path length of a level is the largest shortest-path distance (in steps, not tiles) "
        # "between any two walkable tiles that lie in the same region.\n\n"
        # "Compare the two levels with these rules, in this order:\n"
        # "1. A level without any walkable tile is worse than any level that has at least one.\n"  
        # "2. Regions: the level whose number of regions is closer to exactly 1 is better.\n"
        # "3. Path length: if both levels are equally close to 1 region, the level with the longer path length is better.\n\n"
        # "If both levels are equal under these rules, neither is better and you answer 'none'.\n\n"
        # "Respond STRICTLY in valid JSON with exactly two keys: "
        # "'reasoning' (short explanation: number of regions and path length you found for each level) "
        # "and 'better_level' (value 'A', 'B' or 'none')."
        #)
        system_prompt = (
        "You are an evaluator for 2D grid levels from the PCGRL binary benchmark. "
        "Each level is a 14x14 grid. '#' is a wall, '.' is a walkable tile.\n\n"
        "Definitions:\n"
        "- Two walkable tiles are connected if they are direct horizontal or vertical neighbours (no diagonal moves).\n"
        "- A region is a maximal group of connected walkable tiles.\n"
        "- The path length of a level is the largest shortest-path distance (in steps, not tiles) "
        "between any two walkable tiles that lie in the same region.\n\n"
        "Compare the two levels with these rules:\n"
        "1. A level without any walkable tile is worse than any level that has at least one.\n"
        "2. Otherwise, compute a score for each level: score = path_length - 5 * (regions - 1). "
        "The level with the higher score is better. "
        "This means one region less is worth exactly 5 steps of path length.\n\n"
        "Example: a level with 3 regions and path length 25 has score 25 - 5 * 2 = 15. "
        "A level with 1 region and path length 12 has score 12 - 5 * 0 = 12. "
        "So the first level is better, although it has more regions.\n\n"
        "If both levels have the same score, neither is better and you answer 'none'.\n\n"
        "Respond STRICTLY in valid JSON with exactly two keys: "
        "'reasoning' (short explanation: number of regions, path length and score you found for each level) "
        "and 'better_level' (value 'A', 'B' or 'none')."
        )   
    else: 
        system_prompt = (
        "You are an impartial game design evaluator for 2D tile-based video game levels. "
        "Tile legend: '#' represents solid walls, '.' represents walkable path, 'S' is the start position, and 'E' is the exit/goal. "
        "Your task is to compare two level layouts (Level A and Level B) and decide which one is better designed and playable "
        "(e.g., has a valid clear path from S to E, avoids total blockage, and shows coherent layout). "
        "Respond STRICTLY in valid JSON format with exactly two keys: "
        "'better_level' (value must be either 'A' or 'B') and 'reasoning' (a concise explanation)."
        )

    user_prompt = (
        "Compare the following two levels."
        "\n--- LEVEL A ---"
        f"\n{level_A}\n"
        "\n--- LEVEL B ---"
        f"\n{level_B}\n"
        "\nWhich level is better according to the rules? Respond only with the requested JSON object."
    )

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    }
    if REASONING_EFFORT is not None:
        data["reasoning_effort"] = REASONING_EFFORT
    if REASONING_EFFORT == "none" or REASONING_EFFORT is None:
        data["temperature"] = 0.0

    response = requests.post(url, headers=headers, json=data, timeout=TIMEOUT_SECONDS)
    if not response.ok:
        print(f"Status Code: {response.status_code}")
        print(response.text)
    response.raise_for_status()

    return response    # Response with LLM answer and all meta-data, as well as debug_info

    

def format_answer(raw_response):
    """Takes the raw response from the API call to the LLM and extracts its preferred level and its reasoning.
    preferred_level is 'A', 'B', None (LLM says both are equally good) or 'error' (answer not usable)."""

    llm_text = raw_response.json()["choices"][0]["message"]["content"]  # complete LLM answer

    text = llm_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        return "error", text

    if not isinstance(parsed, dict):
        return "error", text

    reasoning = parsed.get('reasoning')
    answer = parsed.get('better_level')

    if isinstance(answer, str):
        answer = answer.strip().upper()
    if answer in ("A", "B"):
        preferred_level = answer
    elif answer == "NONE":
        preferred_level = None
    else:
        preferred_level = "error"

    return preferred_level, reasoning



def get_gym_level(game=None):
    """Takes a level name (right now only binary-narrow-v0) and generates the level.  
    It returns the level as a matrix, the level converted to a string, and to the game corresponding stats """

    env = gym.make('binary-narrow-v0')
    env.reset()
    raw_env : PcgrlEnv = env.unwrapped

    level_matrix = raw_env._rep._map.copy()  # a 2d-matrix with 1s and 0s
    env.close()

    # Convert numpy array into string
    text = ""
    for row in level_matrix:
        for tile in row:
            if tile == 0:
                text += ". "
            elif tile == 1:
                text += "# "
            #text += str(tile) + " "
        text += "\n"
    short_level_string = text.strip()     # cut last \n 

    env_problem : Problem = raw_env._prob
    tile_types = env_problem.get_tile_types()
    level_string = get_string_map(level_matrix, tile_types)
    env_stats = env_problem.get_stats(level_string)


    # print("DEBUG INFO: ")
    # print("short level string:", short_level_string)
    # print("level string:", level_string)
    # print("level matrix: ", level_matrix)
    # print("env stats: ", env_stats)

    return level_matrix, short_level_string, env_stats


# def evaluate_levels_with_criteria(level_stats_A, level_stats_B, game=None):
#     """Takes the stats of 2 levels (dictionarys) and the name of the game.
#     Calculates which level is the winner based on criteria, specified for the game. 
#     Returns the String 'A', if level A is the winner. 
#     Returns the String 'B', if level B is the winner.
#     Returns None, if both levels are equally good/bad"""
#     if game == "binary" :
#         regions_A = level_stats_A["regions"]
#         regions_B = level_stats_B["regions"]
#         path_length_A = level_stats_A["path-length"]
#         path_length_B = level_stats_B["path-length"]

#         # If a level has 0 regions it's only walls. this shouldn't be allowed.
#         if regions_A == 0:
#             if regions_B == 0:
#                 return None
#             else:
#                 return "B"
#         elif regions_B == 0:
#             return "A"

#         # Levels with regions closer to 1 are better
#         if regions_A < regions_B:
#             return "A"
#         elif regions_A > regions_B:
#             return "B"

#         # If both have same number of regions, the level with longest path_length wins
#         if path_length_A > path_length_B:
#             return "A"
#         elif path_length_A < path_length_B:
#             return "B"
#         else:
#             return None

def evaluate_levels_with_criteria(level_stats_A, level_stats_B, game=None):
    """Takes the stats of 2 levels (dictionarys) and the name of the game.
    Calculates which level is the winner based on criteria, specified for the game. 
    Returns the String 'A', if level A is the winner. 
    Returns the String 'B', if level B is the winner.
    Returns None, if both levels are equally good/bad"""
    if game == "binary" :
        problem = BINARY_PROBLEM

        regions_A = level_stats_A["regions"]
        regions_B = level_stats_B["regions"]

        # If a level has 0 regions it's only walls. this shouldn't be allowed.
        if regions_A == 0:
            if regions_B == 0:
                return None
            else:
                return "B"
        elif regions_B == 0:
            return "A"

        # tests how much better level A is to B
        reward = problem.get_reward(level_stats_A, level_stats_B)

        if reward > 0 :
            return "A"
        elif reward < 0 :
            return "B"
        else:
            return None


def evaluate_LLM_answers(num_tests, out_file = OUT_FILE, print_every_test = True):
    same_answer_count = 0
    error_count = 0
    passed_times = []

    for i in range(1, num_tests + 1):
        if print_every_test:
            print(f"\n================ Test {i} ================")
        matrix_A, lvl_A, stats_A = get_gym_level()
        matrix_B, lvl_B, stats_B = get_gym_level()

        start_time = time.perf_counter()
        response = send_2_levels(lvl_A, lvl_B, game = "binary")
        end_time = time.perf_counter()
        time_gone_by = end_time - start_time
        passed_times.append(time_gone_by)

        LLM_winner, reasoning = format_answer(response)
        criteria_winner = evaluate_levels_with_criteria(stats_A, stats_B, game= "binary")

        data = {
            "level_A" : matrix_A.tolist(),
            "level_B" : matrix_B.tolist(),
            # Transform every stats entry from numpy to int (Could make trouble later on with diffrent stats!!)
            "level_A_stats": {key: int(value) for key, value in stats_A.items()},
            "level_B_stats": {key: int(value) for key, value in stats_B.items()},
            "LLM_label": LLM_winner,
            "criteria_label": criteria_winner,
            "LLM_reasoning": reasoning,
            "LLM_time": time_gone_by,
            "model_name": MODEL,
            "reasoning_effort": REASONING_EFFORT
        }
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

        if print_every_test:
            print("\n----- Level A -----")
            print(f"\n{lvl_A}")
            print("\n----- Level B -----")
            print(f"\n{lvl_B}")
        
            print("\n----- LLM Evaluation -----")
            print(f"Preferred Level : {LLM_winner}")
            print(f"Reasoning       : {reasoning}")
            print(f"Passed time: {time_gone_by}")

            print("\n----- Criteria Evaluation -----")
            print(f"\nStats A: {stats_A}")
            print(f"Stats B: {stats_B}")
            print(f"Preferred Level : {criteria_winner}")

        if LLM_winner == "error":
            error_count += 1
        elif LLM_winner == criteria_winner:
            same_answer_count += 1

    print("\n============= Correspondence of LLM and Code =============")
    print(f"In {same_answer_count} out of {num_tests} comparisons, "
          "the LLM had the same answer as the code evaluation.")
    print(f"{error_count} LLM answers could not be used.")
    print(f"\n\n {(same_answer_count / num_tests) * 100:.1f}% of the comparisons had the same choice")
    print(f"Total passed time: {sum(passed_times)} ")


def test():
    pass


    


if __name__ == "__main__":
    evaluate_LLM_answers(500, out_file="pairs_none.jsonl")
    #test_Connection()
    #test()


