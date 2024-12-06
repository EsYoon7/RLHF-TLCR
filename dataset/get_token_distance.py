import torch
import json
import numpy as np
import os
import ipdb
import pickle
from tqdm import tqdm

from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--process_file', type=str, default='FILE_PATH_HERE')
parser.add_argument('--save_file', type=str, default='FILE_PATH_HERE')
parser.add_argument('--model_name_or_path', type=str, default='MODEL_PATH_OR_NAME_HERE')
args = parser.parse_args()

process_file = args.process_file
save_file = args.save_file
model_name_or_path = args.model_name_or_path

def levenshtein_alignment(reference, prediction):
    """
    Compute the Levenshtein alignment between reference and prediction.
    
    Returns:
        aligned_ref: Aligned reference string
        aligned_pred: Aligned prediction string
    """
    # Initialize the matrix for dynamic programming
    m, n = len(reference), len(prediction)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif reference[i-1] == prediction[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],  # D
                                  dp[i][j-1],  # Insertion
                                  dp[i-1][j-1])  #'S

    # Traceback to get the alignment
    i, j = m, n
    aligned_ref, aligned_pred = [], []
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:  # D
            aligned_ref.append(reference[i-1])
            aligned_pred.append('-')
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:  # Insertion
            aligned_ref.append('-')
            aligned_pred.append(prediction[j-1])
            j -= 1
        else:  # Match or'S
            aligned_ref.append(reference[i-1])
            aligned_pred.append(prediction[j-1])
            i -= 1
            j -= 1

    # Reverse the aligned sequences since we traced back from the end
    return ''.join(aligned_ref[::-1]), ''.join(aligned_pred[::-1])


def extract_errors(aligned_ref, aligned_pred):
    """
    Extract the erroneous characters based on the aligned sequences.
    
    Returns:
        errors: A list of tuples containing (position, reference_char, predicted_char)
    """
    errors = []
    for i, (r_char, p_char) in enumerate(zip(aligned_ref, aligned_pred)):
        if r_char != p_char:
            errors.append((i, r_char, p_char))
    return errors

def process_errors(extracted_errors):
    error_dict = {'subs_in_rejected':{}, 'subs_in_modified':{}, 'ins_in_modified':{}, 'del_in_rejected':{}}

    for error in extracted_errors:
        if error[1] == '-':
            try:
                error_dict['ins_in_modified'][error[2]] += 1
            except:
                error_dict['ins_in_modified'][error[2]] = 1
        elif error[2] == '-':
            try:
                error_dict['del_in_rejected'][error[1]] += 1
            except:
                error_dict['del_in_rejected'][error[1]] = 1
        else:
            try:
                error_dict['subs_in_rejected'][error[1]] += 1
                error_dict['subs_in_modified'][error[2]] += 1
            except:
                error_dict['subs_in_rejected'][error[1]] = 1
                error_dict['subs_in_modified'][error[2]] = 1
    
    return error_dict

def process_errors_idx(extracted_errors):
    error_dict = {'subs_in_rejected':[], 'subs_in_modified':[], 'ins_in_modified':[], 'del_in_rejected':[]}

    for idx, error in enumerate(extracted_errors):
        if error[1] == '-':
            error_dict['ins_in_modified'].append((error[2], error[0]))

        elif error[2] == '-':
            error_dict['del_in_rejected'].append((error[1], error[0]))
        else:
            error_dict['subs_in_rejected'].append((error[1], error[0]))
            error_dict['subs_in_modified'].append((error[2], error[0]))
    
    return error_dict

def alignment2sent(alignment):
    dict_alignment = {}
    sent_idx = 0
    for idx, item in enumerate(alignment):
        if not item == '-':
            dict_alignment[idx] = sent_idx
            sent_idx += 1
    return dict_alignment 
           

def align2sent_error_idx(error_idx_dict, reject_response_idx_dict, aligned_modified_response_idx_dict):
    sent_error_idx_dict = {k : [] for k in error_idx_dict.keys()}
    for error_type, error_idx_list in error_idx_dict.items():
        for error_item in error_idx_list:
            if error_type == 'ins_in_modified':
                sent_error_idx_dict[error_type].append((error_item[0], aligned_modified_response_idx_dict[error_item[1]]))
            elif error_type == 'del_in_rejected':
                sent_error_idx_dict[error_type].append((error_item[0], reject_response_idx_dict[error_item[1]]))
            
            elif error_type == 'subs_in_rejected':
                sent_error_idx_dict[error_type].append((error_item[0], reject_response_idx_dict[error_item[1]]))
            elif error_type == 'subs_in_modified':
                sent_error_idx_dict[error_type].append((error_item[0], aligned_modified_response_idx_dict[error_item[1]]))
    return sent_error_idx_dict

def label_elements_with_changes(A, B):
    # Initialize the DP matrix with operation counts and back pointers
    dp = [[(0, None) for _ in range(len(B) + 1)] for _ in range(len(A) + 1)]

    # Initialize base cases
    for i in range(1, len(A) + 1):
        dp[i][0] = (i, 'D')
    for j in range(1, len(B) + 1):
        dp[0][j] = (j, 'A')

    # Fill DP matrix
    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            if A[i-1] == B[j-1]:
                dp[i][j] = (dp[i-1][j-1][0], 'U')
            else:
                operations = [
                    (dp[i-1][j][0] + 1, 'D'),
                    (dp[i][j-1][0] + 1, 'A'),
                    (dp[i-1][j-1][0] + 1, 'S')
                ]
                dp[i][j] = min(operations, key=lambda x: x[0])

    # Backtrack to label elements
    labeled_A, labeled_B = [], []
    i, j = len(A), len(B)
    while i > 0 or j > 0:
        operation = dp[i][j][1]
        if operation == 'D':
            labeled_A.insert(0, 'D')
            i -= 1
        elif operation == 'A':
            labeled_B.insert(0, 'A')
            j -= 1
        elif operation == 'S':
            labeled_A.insert(0, 'S')
            labeled_B.insert(0, 'S')
            i -= 1
            j -= 1
        elif operation == 'U':
            labeled_A.insert(0, 'U')
            labeled_B.insert(0, 'U')
            i -= 1
            j -= 1

    # Handling cases where elements are added to B at the beginning
    while j > 0:
        labeled_B.insert(0, 'A')
        j -= 1

    # Handling cases where elements are deleted from A at the beginning
    while i > 0:
        labeled_A.insert(0, 'D')
        i -= 1

    return labeled_A, labeled_B

with open(process_file, 'r') as f:
    text_data = json.load(f)

output_path = save_file

processed_text_data_json = []
processed_text_data_dict = {k : [] for k in text_data[0].keys()}
processed_text_data_dict['response'] = []
processed_text_data_dict['rejected_token_edit'] = []
processed_text_data_dict['modified_token_edit'] = []

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

for idx, item in enumerate(tqdm(text_data)):
    reject_response = item['reject_response']
    raw_response = item['raw_response']
    modification = raw_response.split('<|Start Of Modified Response|>\n')[-1]
    
    reject_token = tokenizer(reject_response, add_special_tokens=False)['input_ids']
    modification_token = tokenizer(modification, add_special_tokens=False)['input_ids']
    
    rejected_token_edit_info, modified_token_edit_info = label_elements_with_changes(reject_token, modification_token)


    item['response'] = modification
    item['rejected_token_edit'] = rejected_token_edit_info
    item['modified_token_edit'] = modified_token_edit_info

    processed_text_data_json.append(item)
    for k, v in item.items():
        processed_text_data_dict[k].append(v)

    if idx % 100 == 0:
        with open(os.path.join(output_path, 'eval_data_temp.json'), 'w') as f:
            json.dump(processed_text_data_json, f)

        with open(os.path.join(output_path, 'eval_data_temp.pkl'), 'wb') as f:
            pickle.dump(processed_text_data_dict, f)

output_fname_json = os.path.join(output_path, 'eval_data_final.json')
output_fname_pkl = os.path.join(output_path, 'eval_data_final.pkl')

with open(output_fname_json, 'w') as f:
    json.dump(processed_text_data_json, f)

with open(output_fname_pkl, 'wb') as f:
    pickle.dump(processed_text_data_dict, f)

ipdb.set_trace()