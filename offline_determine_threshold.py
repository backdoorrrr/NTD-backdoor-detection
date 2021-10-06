import os
import cv2
import csv
import random
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
import utils.utils as utils
import matplotlib.pyplot as plt
from scipy.stats import gamma, loggamma
from net.inception import InceptionResNetV1


class FeatureExtractor:

    def __init__(self):

        # Loading FaceNet, convert face sample into 128-dimensional vector
        self.facenet_model = InceptionResNetV1()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

    def cal_pic_feature(self, pic_path):
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize input sample to 160*160*3
        crop_img = np.expand_dims(cv2.resize(img, (160, 160)), 0)
        # Extract 128-dimensional feature vector
        face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)

        return face_encoding


# Randomly select one person for global
def choose_person(dir):
    persons_list = os.listdir(dir)

    chosen_person = random.sample(persons_list, k=1)[0]
    # person = 'Woody_Allen'
    return chosen_person


# Calculate Intra/Inter Similarity
def cal_inter_intra_sim(person):

    sample_feature = []
    person_dir = os.path.join(dataset_dir, person)
    samples_list = os.listdir(person_dir)
    # Randomly select a sample x and a set of samples of the comarison set{x_1, x_2, ..., x_n}
    random_samples_list = random.sample(samples_list, k=comparison_set_n+1)
    for sample in random_samples_list:
        sample_path = os.path.join(person_dir, sample)
        sample_feature.append(FE.cal_pic_feature(sample_path))

    # a random person different from x
    inter_sample_feature = []
    inter_persons = list(set(os.listdir(dataset_dir)) - set([person]))
    random_inter_person = random.sample(inter_persons, k=1)[0]

    # samples of inter class{y_1, y_2, ..., y_n}
    inter_person_dir = os.path.join(dataset_dir, random_inter_person)
    inter_samples_list = os.listdir(inter_person_dir)
    random_inter_samples_list = random.sample(inter_samples_list, k=comparison_set_n)

    for sample in random_inter_samples_list:
        sample_path = os.path.join(inter_person_dir, sample)
        inter_sample_feature.append(FE.cal_pic_feature(sample_path))

    # Average Similarity
    intra_sim, inter_sim = 0, 0
    for i in range(comparison_set_n):
        intra_sim += sim(similarity_metric, sample_feature[0], sample_feature[i + 1])
        inter_sim += sim(similarity_metric, sample_feature[0], inter_sample_feature[i])
    intra_sim = intra_sim / comparison_set_n
    inter_sim = inter_sim / comparison_set_n

    return float('%.5f' % intra_sim), float('%.5f' % inter_sim)


# Calculate All(1,000 rounds) Similarity
def cal_sim_all(person):
    inter_sim, intra_sim = [], []

    for _ in tqdm(range(rounds)):
        intra_temp, inter_temp = cal_inter_intra_sim(person)
        intra_sim.append(intra_temp)
        inter_sim.append(inter_temp)

    return intra_sim, inter_sim


# Calculate Similarity
def sim(sim_type, feature_a, feature_b):

    sim_a_b = utils.face_cosdistance(sim_type, feature_a, feature_b)

    return sim_a_b

# Determine Threshold
def dtm_threshold(intra_sim, inter_sim):
    # Fit Curve
    arg_inter = gamma.fit(inter_sim)
    arg_intra = loggamma.fit(intra_sim)

    # right = loggamma.ppf(percent, c=arg_intra[0], loc=arg_intra[1], scale=arg_intra[2])
    # left = gamma.ppf(0.9 + percent, a=arg_inter[0], loc=arg_inter[1], scale=arg_inter[2])

    threshold = loggamma.ppf(preset_frr, c=arg_intra[0], loc=arg_intra[1], scale=arg_intra[2])
    far_offline = 1 - gamma.cdf(threshold, a=arg_inter[0], loc=arg_inter[1], scale=arg_inter[2])



    paint(intra_sim, inter_sim, threshold)
    return threshold, far_offline


def paint(intra_sim, inter_sim, threshold):
    plt.axvline(threshold, color='red', label='Threshold')
    sns.distplot(intra_sim, label='Intra Similarity', color="b", bins=500, kde=False, fit=loggamma,
                 fit_kws={'color': 'b', 'label': 'Intra Fit Curve'})
    sns.distplot(inter_sim, label='Inter Similarity', color="g", bins=500, kde=False, fit=gamma,
                 fit_kws={'color': 'g', 'label': 'Inter Fit Curve'})
    plt.legend(fontsize=15)
    plt.xlabel('Similarity', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig('./plot/offline_plot/%s_%s_%s_%s.png' % (
        person if not global_trd else 'global', comparison_set_n, similarity_metric,  preset_frr))
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--person', help="Input a person name (e.g. Woody_Allen) or global", required=True)
    parser.add_argument('-s', '--similarity_metric', default='tanimoto', choices=['cosine', 'pearson', 'tanimoto'])
    parser.add_argument('-n', '--comparison_set_n', default=3, type=int)
    parser.add_argument('-r', '--preset_frr', default=0.005, type=float)
    args = parser.parse_args()


    person = args.person
    similarity_metric = args.similarity_metric
    comparison_set_n = args.comparison_set_n
    preset_frr = args.preset_frr

    rounds = 1000
    dataset_dir = './dataset/offline_facescrub_dataset'
    offline_csv = './result/offline_csv_result.csv'
    global_trd = False if not args.person == 'global' else True

    FE = FeatureExtractor()

    # Global Threshold
    if person == 'global':

        intra_sim, inter_sim = [], []
        print("Offline: Calculate all inter and intra similarity, person name=global")
        for _ in tqdm(range(rounds)):
            person = choose_person(dataset_dir)
            intra_temp, inter_temp = cal_inter_intra_sim(person)
            intra_sim.append(intra_temp)
            inter_sim.append(inter_temp)
    # Per Class Individual Threshold
    elif person in os.listdir(dataset_dir):
        print("Offline: Calculate all inter and intra similarity, person name=%s" % person)
        intra_sim, inter_sim = cal_sim_all(person)
    else:
        print("%s doesn't exist!" % person)
        exit(0)

    threshold, far_offline = dtm_threshold(intra_sim, inter_sim)
    print('threshold = ', threshold, 'far_offline = ', far_offline)

    with open(offline_csv, 'a', newline='') as file:
        csv_file = csv.writer(file)
        datas = [person if not global_trd else 'global', str(preset_frr), str(far_offline), str(threshold),
                 similarity_metric, str(comparison_set_n)]
        csv_file.writerow(datas)