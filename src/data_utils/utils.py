import re

# ascii code, used to delete Chinese punctuation
CHN_PUNC_LIST = [183, 215, 8212, 8216, 8217, 8220, 8221, 8230,
                 12289, 12290, 12298, 12299, 12302, 12303, 12304, 12305,
                 65281, 65288, 65289, 65292, 65306, 65307, 65311]
CHN_PUNC_SET = set(CHN_PUNC_LIST)

MANDARIN_INITIAL_LIST = ["b", "ch", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "sh", "s", "t",
                         "x", "zh", "z"]

# prosody phone list
CHN_PHONE_PUNC_LIST = ['sp2', 'sp1', 'sil']
# erhua phoneme
CODE_ERX = 0x513F


def _update_insert_pos(old_pos, pylist):
    new_pos = old_pos + 1
    i = new_pos
    while i < len(pylist) - 1:
        # if the first letter is upper, then this is the phoneme of English letter
        if pylist[i][0].isupper():
            i += 1
            new_pos += 1
        else:
            break
    return new_pos


def _pinyin_preprocess(line, words):
    if line.find('.') >= 0:
        # remove '.' in English letter phonemes, for example: 'EH1 F . EY1 CH . P IY1'
        py_list = line.replace('/', '').strip().split('.')
        py_str = ''.join(py_list)
        pinyin = py_str.split()
    else:
        pinyin = line.replace('/', '').strip().split()

    # now the content in pinyin like: ['OW1', 'K', 'Y', 'UW1', 'JH', 'EY1', 'shi4', 'yi2', 'ge4']
    insert_pos = _update_insert_pos(-1, pinyin)
    i = 0
    while i < len(words):
        if ord(words[i]) in CHN_PUNC_SET:
            i += 1
            continue
        if words[i] == '#' and '1' <= words[i + 1] <= '4':
            if words[i + 1] == '1':
                pass
            else:
                if words[i + 1] == '2':
                    pinyin.insert(insert_pos, 'sp2')
                if words[i + 1] == '3':
                    pinyin.insert(insert_pos, 'sp2')
                elif words[i + 1] == '4':
                    pinyin.append('sil')
                    break
                insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 2
        elif ord(words[i]) == CODE_ERX:
            if pinyin[insert_pos - 1].find('er') != 0:  # erhua
                i += 1
            else:
                insert_pos = _update_insert_pos(insert_pos, pinyin)
                i += 1
        # skip non-mandarin characters, including A-Z, a-z, Greece letters, etc.
        elif ord(words[i]) < 0x4E00 or ord(words[i]) > 0x9FA5:
            i += 1
        else:
            insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 1
    return pinyin


def _pinyin_2_initialfinal(py):
    """
    used to split pinyin into intial and final phonemes
    """
    if py[0] == 'a' or py[0] == 'e' or py[0] == 'E' or py[0] == 'o' or py[:2] == 'ng' or \
            py[:2] == 'hm':
        py_initial = ''
        py_final = py
    elif py[0] == 'y':
        py_initial = ''
        if py[1] == 'u' or py[1] == 'v':
            py_final = list(py[1:])
            py_final[0] = 'v'
            py_final = ''.join(py_final)
        elif py[1] == 'i':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'i'
            py_final = ''.join(py_final)
    elif py[0] == 'w':
        py_initial = ''
        if py[1] == 'u':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'u'
            py_final = ''.join(py_final)
    else:
        init_cand = ''
        for init in MANDARIN_INITIAL_LIST:
            init_len = len(init)
            init_cand = py[:init_len]
            if init_cand == init:
                break
        if init_cand == '':
            raise Exception('unexpected')
        py_initial = init_cand
        py_final = py[init_len:]
        if py_initial in {'j', 'q', 'x'} and py_final[0] == 'u':
            py_final = list(py_final)
            py_final[0] = 'v'
            py_final = ''.join(py_final)
    if py_final[-1] == '6':
        py_final = py_final.replace('6', '2')
    return py_initial, py_final


def is_all_eng(words):
    # if include mandarin
    for word in words:
        if 0x4E00 <= ord(word) <= 0x9FA5:
            return False
    return True


def pinyin_2_phoneme(pinyin_line, words):
    # chn or chn+eng
    sent_phoneme = ['sp1']
    if not is_all_eng(words):
        sent_py = _pinyin_preprocess(pinyin_line, words)
        for py in sent_py:
            if py[0].isupper() or py in CHN_PHONE_PUNC_LIST:
                sent_phoneme.append(py)
            else:
                initial, final = _pinyin_2_initialfinal(py)
                if initial == '':
                    sent_phoneme.append(final)
                else:
                    sent_phoneme.append(initial)
                    sent_phoneme.append(final)
    else:
        wordlist = words.split(' ')
        word_phonelist = pinyin_line.strip().split('/')
        assert (len(word_phonelist) == len(wordlist))
        i = 0
        while i < len(word_phonelist):
            phone = re.split(r'[ .]', word_phonelist[i])
            for p in phone:
                if p:
                    sent_phoneme.append(p)
            if '/' in wordlist[i]:
                sent_phoneme.append('sp2')
            elif '%' in wordlist[i]:
                if i != len(word_phonelist) - 1:
                    sent_phoneme.append('sp2')
                else:
                    sent_phoneme.append('sil')
            i += 1
    return ' '.join(sent_phoneme)
