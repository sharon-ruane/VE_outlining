import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from keras.models import model_from_json
from U_NET_training_functions import emb_image_batch_generator_v2


def predict_from_folder(model_dir, batch):
    json_file = open(os.path.join(model_dir, "model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(model_dir, "weights.h5"))
    outlines_pred = model.predict(batch, batch_size=batch.shape[0],
                                  verbose=1, steps=None)
    print(batch.shape[0])
    y_pred = outlines_pred.ravel()
    return y_pred


batch_size = 100
im_size = (96, 96)
data_folder = "/home/iolie/PhD_Thesis_Data/epithelial_cell_border_identification"
saved_test = 'saved_testset.json'
test_data = json.load(open(saved_test))
save_dir = "/home/iolie/PycharmProjects/THESIS/ROC plots"

correct_models = [
    'FINAL_TEST_UNET_1_(min size 52)/titletraining_weightsatloss_0.36',
    'FINAL_TEST_UNET_1_(min size 72)/titletraining_weightsatloss_0.35',
    'FINAL_TEST_UNET_2_(min size 52)/titletraining_weightsatloss_0.46',
    'FINAL_TEST_UNET_2_(min size 72)/titletraining_weightsatloss_0.36',
    'FINAL_TEST_UNET_3_(min size 52)/titletraining_weightsatloss_0.37',
    'FINAL_TEST_UNET_3_(min size 72)/titletraining_weightsatloss_0.36',
    'FINAL_TEST_UNET_4_(min size 52)/titletraining_weightsatloss_0.35',
    'FINAL_TEST_UNET_4_(min size 72)/titletraining_weightsatloss_0.35',
    'FINAL_TEST_UNET_5_(min size 52)/titletraining_weightsatloss_0.36',
    'FINAL_TEST_UNET_5_(min size 72)/titletraining_weightsatloss_0.36',
    'FINAL_TEST_UNET_6_(min size 52)/titletraining_weightsatloss_0.35',
    'FINAL_TEST_UNET_6_(min size 72)/titletraining_weightsatloss_0.35']

batch_gen = emb_image_batch_generator_v2(data_folder, test_data, batch_size,
                                         im_size, MIN_SIZE=im_size[0])
raw_image_batch, ground_truth_batch = next(batch_gen)


#
# print zip(correct_models[0::2], correct_models[1::2])
# counter = 1
# for i,k in zip(correct_models[0::2], correct_models[1::2]):
#     print "ROC chart for: ", str(i), '+', str(k)
#     i_label = "U-Net" + i.split("_")[3] + " " + (i.split("_")[4]).split("/")[0]
#     k_label = "U-Net" + k.split("_")[3] + " " + (k.split("_")[4]).split("/")[0]
#     y_pred_model_1 = predict_from_folder(i, raw_image_batch)
#     fpr_model_1, tpr_model_1, thresholds_model_1 = roc_curve(ground_truth_batch.ravel(), y_pred_model_1)
#
#     y_pred_model_2 = predict_from_folder(k, raw_image_batch)
#     fpr_model_2, tpr_model_2, thresholds_model_2 = roc_curve(ground_truth_batch.ravel(), y_pred_model_2)
#     auc_model_1 = auc(fpr_model_1, tpr_model_1)
#     auc_model_2 = auc(fpr_model_2, tpr_model_2)
#
#     plt.clf()
#     plt.figure(1)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr_model_1, tpr_model_1,label= i_label + '(area = {:.3f})'.format(auc_model_1))
#     plt.plot(fpr_model_2, tpr_model_2,label= k_label + '(area = {:.3f})'.format(auc_model_2))
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curves:' + "Model " + str(counter) + " and Model " + str(counter + 1))
#     plt.legend(loc='best')
#     plt.savefig(os.path.join(save_dir, "U-Net" + i.split("_")[3] + (i.split("_")[4]).split("/")[0] + " + " + (k.split("_")[4]).split("/")[0]))
#     counter = counter + 2

FPRs = []
TPRs = []
THRESHOLDS = []
AUCs = []
labels = []

for i in (correct_models):
    print(i)
    i_label = "U-Net" + i.split("_")[3] + " " + (i.split("_")[4]).split("/")[0]
    print(i_label)
    labels.append(i_label)

    y_pred_model_1 = predict_from_folder(i, raw_image_batch)
    fpr_model_1, tpr_model_1, thresholds_model_1 = roc_curve(ground_truth_batch.ravel(), y_pred_model_1)
    auc_model_1 = auc(fpr_model_1, tpr_model_1)
    FPRs.append(fpr_model_1)
    TPRs.append(tpr_model_1)
    THRESHOLDS.append(thresholds_model_1)
    AUCs.append(auc_model_1)

plt.clf()
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(len((correct_models[0::2]))):
    plt.plot(FPRs[i], TPRs[i],label= labels[i] + ' (area = {:.3f})'.format(AUCs[i]))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves')
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, "All twelve"))











# predict_from_folder(correct_models[0], raw_image_batch)

# FPRs = []
# TPRs = []
# THRESHOLDS = []
# AUCs = []


# y_pred_model_1 = predict_from_folder(correct_models[0], raw_image_batch)
# fpr_model_1, tpr_model_1, thresholds_model_1 = roc_curve(ground_truth_batch.ravel(), y_pred_model_1)
#
# y_pred_model_2 = predict_from_folder(correct_models[1], raw_image_batch)
# fpr_model_2, tpr_model_2, thresholds_model_2 = roc_curve(ground_truth_batch.ravel(), y_pred_model_2)
#
# auc_model_1 = auc(fpr_model_1, tpr_model_1)
# auc_model_2 = auc(fpr_model_2, tpr_model_2)
# print("auc_model_1: " , auc_model_1)
# print("auc_model_2: " , auc_model_2)











# Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_model_1, tpr_model_1, label='M1 (area = {:.3f})'.format(auc_model_1))
# plt.plot(fpr_model_2, tpr_model_2, label='M2 (area = {:.3f})'.format(auc_model_1))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()
#
#
#

# for i in range(len(correct_models)):
#     y_pred_model = predict_from_folder(correct_models[i], raw_image_batch)
#     fpr_model_i, tpr_model_i, thresholds_model_i = roc_curve(ground_truth_batch.ravel(), y_pred_model)
# 	auc_model_i = auc(fpr_model_i, tpr_model_i)
# 	FPRs.append(fpr_model_i)
# 	TPRs.append(tpr_model_i)
# 	THRESHOLDS.append(thresholds_model_i)
# 	AUCs.append(auc_model_i)

# print(FPRs)
# print(FPRs[0:11:2])
# for i in range(len(correct_models)//2):
# 	FPRs[0:11:2]
