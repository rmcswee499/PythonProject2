import numpy as np
import scipy.stats as stats
import string

# global constants
rcp_unit_flash_num = 12
rcp_char_size = 36
rcp_screen = np.reshape(np.arange(rcp_char_size), [6, 6]) + 1
stimulus_group_set = []
for row_iter in range(6):
    stimulus_group_set.append(rcp_screen[row_iter, :])
for col_iter in range(6):
    stimulus_group_set.append(rcp_screen[:, col_iter])
eeg_rcp_array = list(string.ascii_uppercase)
string_digits = list(string.digits)
string_digits.remove('0')
string_digits.append('0')
eeg_rcp_array.extend(string_digits)
trn_seq_size=15


def convert_raw_char_to_alphanumeric_stype(input_char):
    input_char = np.char.strip(np.array([input_char[i_iter][0][0] for i_iter in range(len(input_char))]))
    input_char[input_char == '<BS>'] = 8
    input_char[input_char == ''] = 0
    input_char = input_char.tolist()
    return input_char


def compute_summary_parameter(input_score, input_type):
    input_score = np.log(input_score)
    mu_tar = np.mean(input_score[input_type == 1])
    mu_ntar = np.mean(input_score[input_type != 1])
    std_common = np.std(input_score)
    return mu_tar, mu_ntar, std_common


def _ml_predict_letter_likelihood_unit(
        char_prob, stimulus_score, stimulus_code, mu_tar, mu_ntar, std_common, unit_stimulus_set
):
    """
    Apply the bayesian naive dynamic stopping criterion
    :param char_prob:
    :param stimulus_score:
    :param stimulus_code:
    :param mu_tar:
    :param mu_ntar:
    :param std_common:
    :param unit_stimulus_set:
    :return:
    """
    char_prob_post = np.copy(np.log(char_prob))
    for s_id in range(rcp_unit_flash_num):
        for char_id in range(1, rcp_char_size + 1):
            if char_id in unit_stimulus_set[stimulus_code[s_id]-1]:
                char_prob_post[char_id-1] = char_prob_post[char_id-1] + \
                                         stats.norm.logpdf(stimulus_score[s_id], loc=mu_tar, scale=std_common)
            else:
                char_prob_post[char_id-1] = char_prob_post[char_id-1] + \
                                         stats.norm.logpdf(stimulus_score[s_id], loc=mu_ntar, scale=std_common)
    char_prob_post = char_prob_post - np.max(char_prob_post)
    char_prob_post = np.exp(char_prob_post)
    char_prob_post = char_prob_post / np.sum(char_prob_post)
    return char_prob_post


def ml_predict_letter_likelihood(
        stimulus_score, stimulus_code, letter_dim, repet_pred,
        mu_tar, mu_ntar, std_common, unit_stimulus_set, letter_table_ls
):
    stimulus_score = np.reshape(stimulus_score, [letter_dim, repet_pred, rcp_unit_flash_num])
    stimulus_code = np.reshape(stimulus_code, [letter_dim, repet_pred, rcp_unit_flash_num])
    char_prob_mat = np.zeros([letter_dim, repet_pred+1, rcp_char_size]) + 1 / rcp_char_size
    for letter_id in range(letter_dim):
        for seq_id in range(repet_pred):
            char_prob_mat[letter_id, seq_id+1, :] = _ml_predict_letter_likelihood_unit(
                char_prob_mat[letter_id, seq_id, :], stimulus_score[letter_id, seq_id, :],
                stimulus_code[letter_id, seq_id, :], mu_tar, mu_ntar, std_common, unit_stimulus_set
            )
    char_prob_mat = char_prob_mat[:, 1:, :]
    argmax_prob_mat = np.argmax(char_prob_mat, axis=-1)
    char_max_mat = np.zeros_like(argmax_prob_mat).astype('<U5')
    for letter_id in range(letter_dim):
        for seq_id in range(repet_pred):
            char_max_mat[letter_id, seq_id] = letter_table_ls[argmax_prob_mat[letter_id, seq_id]]
    return char_max_mat, char_prob_mat


def streamline_predict(
        y_trn, eeg_type, eeg_code, char_size, seq_size_num,
        stimulus_group_set_inner, eeg_rcp_array_inner
):
    y_trn = y_trn[:, 0]
    mu_tar, mu_ntar, std_common = compute_summary_parameter(
        y_trn, eeg_type
    )
    letter_mat, letter_prob_mat = ml_predict_letter_likelihood(
        y_trn, eeg_code, char_size, seq_size_num,
        mu_tar, mu_ntar, std_common,
        stimulus_group_set_inner, eeg_rcp_array_inner
    )
    return letter_mat, letter_prob_mat