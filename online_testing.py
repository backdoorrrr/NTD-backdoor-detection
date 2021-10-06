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
from offline_determine_threshold import FeatureExtractor, sim


# Choose person
def choose_person(dir):
    persons_list = os.listdir(dir)
    chosen_person = random.sample(persons_list, k=1)[0]

    return chosen_person


# Add trigger
def add_trigger(trigger_img, person_sample):
    trigger = cv2.imread(trigger_img)
    trigger_rs = cv2.resize(trigger, (person_sample.shape[1], person_sample.shape[0]))
    add_trigger_face = cv2.addWeighted(person_sample, 1, trigger_rs, 1, 0)

    return add_trigger_face


# Calculate feature vector
def cal_pic_feature(trigger, pic_path):

    data = cv2.imread(pic_path)
    if trigger:
        data = add_trigger(trigger_img, data)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    crop_img = np.expand_dims(cv2.resize(data, (160, 160)), 0)
    face_encoding = utils.calc_128_vec(FE.facenet_model, crop_img)

    return face_encoding


#Calculate intra/inter similarity at once
def cal_inter_intra_sim(person):

    # Randomly select a sample x of a person
    person_dir = os.path.join(testing_dataset_dir, person)
    samples_list = os.listdir(person_dir)
    random_sample = random.sample(samples_list, k=1)[0]
    sample_path = os.path.join(person_dir, random_sample)
    sample_feature = cal_pic_feature(False, sample_path)

    # A sample of one other person 'y', is different from class x, added to Trigger
    inter_persons = list(set(os.listdir(testing_dataset_dir)) - set([person]))
    random_inter_person = random.sample(inter_persons, k=1)[0]
    inter_person_dir = os.path.join(testing_dataset_dir, random_inter_person)
    inter_samples_list = os.listdir(inter_person_dir)
    random_inter_sample = random.sample(inter_samples_list, k=1)[0]
    sample_path = os.path.join(inter_person_dir, random_inter_sample)
    inter_sample_feature = cal_pic_feature(True, sample_path)

    # {x_1, x_2, ..., x_n} Comparison set of class z
    val_sample_feature = []
    val_person_dir = os.path.join(reserved_val_dataset_dir, person)
    val_samples_list = os.listdir(val_person_dir)
    random_val_samples_list = random.sample(val_samples_list, k=comparison_set_n)

    for val_sample in random_val_samples_list:
        val_sample_path = os.path.join(val_person_dir, val_sample)
        val_sample_feature.append(cal_pic_feature(False, val_sample_path))

    # Average Similarity
    intra_sim, inter_sim = 0, 0
    for i in range(comparison_set_n):
        intra_sim += sim(similarity_metric, sample_feature, val_sample_feature[i])
        inter_sim += sim(similarity_metric, inter_sample_feature, val_sample_feature[i])
    intra_sim = intra_sim / comparison_set_n
    inter_sim = inter_sim / comparison_set_n

    return float('%.5f' % intra_sim), float('%.5f' % inter_sim)


# Calculate all similarity
def cal_sim_all(person):
    inter_sim, intra_sim = [], []

    for _ in tqdm(range(rounds)):
        intra_temp, inter_temp = cal_inter_intra_sim(person)
        intra_sim.append(intra_temp)
        inter_sim.append(inter_temp)

    return intra_sim, inter_sim


# Calculate online FAR/FRR
def cal_far_frr(threshold, intra_sim, inter_sim):

    far_sum, frr_sum = 0, 0
    for sim in intra_sim:
        if sim < threshold:
            frr_sum += 1
    for sim in inter_sim:
        if sim > threshold:
            far_sum += 1

    return far_sum / len(intra_sim), frr_sum / len(inter_sim)


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

    plt.savefig(
        './plot/online_plot/%s_%s_%s_%s_%s.png' % (
            person if not global_trd else 'global', comparison_set_n, similarity_metric, preset_frr, trigger_img[-5]))
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
    trigger_img = './trigger/TriggerA.jpg'
    offline_csv = './result/offline_csv_result.csv'
    testing_dataset_dir = './dataset/online_facescrub_dataset'
    reserved_val_dataset_dir = './dataset/offline_facescrub_dataset'
    global_trd = False if not args.person == 'global' else True

    with open(offline_csv, 'r') as file:
        csv_file = csv.DictReader(file)
        try:
            for row in csv_file:
                if person == row['Name'] and preset_frr == float(row['Preset_FRR']) and \
                        similarity_metric == row['Similarity_Metric'] and comparison_set_n == int(row['Comparison_Set_N']):
                    _, _, far_offline, threshold, _, _ = row.values()
                    far_offline, threshold = float(far_offline), float(threshold)
                    break
            if far_offline == '':
                pass
        except NameError:
            print("No matching threshold was found.")
            exit(0)

    FE = FeatureExtractor()

    if person == 'global':

        intra_sim, inter_sim = [], []
        print("Online: Calculate all inter and intra similarity, person name=global")
        for _ in tqdm(range(rounds)):
            person = choose_person(testing_dataset_dir)
            intra_temp, inter_temp = cal_inter_intra_sim(person)
            intra_sim.append(intra_temp)
            inter_sim.append(inter_temp)
    elif person in os.listdir(reserved_val_dataset_dir):
        print("Online: Calculate all inter and intra similarity, person name=%s" % person)
        intra_sim, inter_sim = cal_sim_all(person)
    else:
        print("%s doesn't exist!" % person)
        exit(0)

    far, frr = cal_far_frr(threshold, intra_sim, inter_sim)
    print('far = ', far, 'frr = ', frr)
    paint(intra_sim, inter_sim, threshold)

    # save result
    result_file = './result/result.txt'
    with open(result_file, "a") as file:
        file.write(
            "Experiment Settings & Result : {person_name=%s, samples_num=%s, frr_percent=%s, far_offline=%s,"
            " threshold=%s, sim_metric=%s, far=%s, frr=%s, trigger=%s}"
            % (person if not global_trd else 'global', comparison_set_n, preset_frr, far_offline, threshold,
                similarity_metric, far, frr, trigger_img[-5]) + '\n')