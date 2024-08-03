from PymoNNto import *

#each time Experiment_char.py is executed, a new folder is created (Data/StorageManager/Experiment_char/Experiment_char_????) containing the involved parameters and the resulting output.
#This script iterates over all results in Data/StorageManager/Experiment_char/, calculates a score and calculates the mean, standard deviation and variance of all this results.

def get_nr_right_word_score(txt, words):
    txt_words = txt.replace('.', '').split(' ')[1:-1]
    right = 0
    wrong = 0
    for tw in txt_words:
        if tw in words:
            right += 1
        else:
            wrong += 1

    return right/(right+wrong)*100

smg = StorageManagerGroup('Experiment_char')

scores = []
for sm in smg:
    txt = sm.load_param('text')
    if txt is not None:
        score = get_nr_right_word_score(txt, ['fox', 'eats', 'meat', 'boy', 'drinks', 'juice', 'penguin', 'likes', 'ice']) #[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.']
        scores.append(score)

print(scores)
print(np.mean(scores), np.std(scores), np.var(scores))

plt.scatter(np.zeros(len(scores)), scores)
plt.scatter([0], [np.mean(scores)])
plt.ylim(0-1, 100+1)
plt.show()