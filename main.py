import os
from torch.utils.data import DataLoader
from dataset import VideoDataset
import random
import scipy.stats as stats
import torchvision.models as ptmodels
from models.fc_finetune import fc_finetune
from models.LSTM_final import LSTM_final
from models.skill_classifier import skill_classifier
from models.MSTCN import *
from opts import *
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import recall_score

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def train(train_dataloader, optimizer, criterions, epoch):
    print('Training starts ...')
    criterion_final_score = criterions['criterion_final_score']; penalty_final_score = criterions['penalty_final_score']
    criterion_incremental_subscore = criterions['criterion_incremental_subscore']

    if with_skill_classification:
        criterion_single_label_classifier = criterions['criterion_single_label_classifier']
        criterion_multi_label_classifier = criterions['criterion_multi_label_classifier']
    if with_gesture_recognition:
        criterion_seg_ce = criterions['criterion_seg_ce']
        criterion_seg_mse = criterions['criterion_seg_mse']

    model_lstm.train()

    model_finetune.train()


    if with_skill_classification:
        model_classifier.train()
    if with_gesture_recognition:
        model_seg.train()

    iteration = 0
    for _, data in enumerate(train_dataloader):
        true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
        if with_skill_classification:
            true_gesimp = torch.from_numpy(np.array(data['label_gesimp'])).float().unsqueeze_(0).cuda()
            true_sp_skill = data['label_sp_skill'].type(torch.LongTensor).cuda()
        if with_gesture_recognition:
            true_seg = data['label_seg'].cuda()
            true_seg_mask = data['label_seg_mask'].cuda()
            true_impre = data['label_impre']

        clip_feats = torch.Tensor([]).cuda()

        if feature_extractor == 'C3D':
            clip_feats = torch.load(features_path + data['name'][0] + '.pt')
            clip_feats_long = np.load(features_full_path + data['name'][0] + '.npy')
        elif feature_extractor == 'resnet-101':
            clip_feats = clip_feats_long = torch.load(features_path + data['name'][0] + '.pt')
            clip_feats_long = clip_feats_long.squeeze(0).transpose(0,1).cpu().numpy()

        # LSTM
        if mode == 'IMTL-AGF':
            seg_recognition = true_seg.cpu().numpy()[0].tolist()

            cursor = 0
            cursor_list = []

            for _, item in enumerate(seg_recognition):
                if _ != len(seg_recognition) - 1 and item != seg_recognition[_ + 1]:
                    cursor_list.append((cursor, _))
                    cursor = _ + 1
                if _ == len(seg_recognition) - 1:
                    cursor_list.append((cursor, _))

            # form ground-truth intermediate scores based on additional annotations of gestures' skill levels
            true_impre = preprocessing.normalize(np.array([true_impre]), norm='l1', axis=1)[0].tolist()

            acc_sum = 0.0
            for _, item in enumerate(cursor_list):
                acc_sum += true_impre[_]
                if finetune_backbone:
                    pred_incre_score, last_output = model_lstm(model_finetune(clip_feats[:, 0:int(item[1] / 8) + 1, :]))
                else:
                    pred_incre_score, last_output = model_lstm(clip_feats[:, 0:int(item[1]/8) + 1, :])
                tmp_loss = torch.zeros(3)
                if with_skill_classification:
                    pred_gesimp = model_classifier(last_output)
                    loss_gesimp = criterion_multi_label_classifier(pred_gesimp, true_gesimp)
                    tmp_loss[0] = loss_gesimp

                true_incre_score = true_final_score * acc_sum
                loss_incre_score = (criterion_final_score(pred_incre_score, true_incre_score)
                                    + penalty_final_score(pred_incre_score, true_incre_score))

                tmp_loss[1] = loss_incre_score

                if _ > 0:
                    if finetune_backbone:
                        pred_incre_score_last, last_output_last = model_lstm(model_finetune(clip_feats[:, 0:int(cursor_list[_ - 1][1] / 8) + 1, :]))
                    else:
                        pred_incre_score_last, last_output_last = model_lstm(clip_feats[:, 0:int(cursor_list[_-1][1]/8)+1, :])

                    loss_rank = criterion_incremental_subscore(pred_incre_score.cpu(), pred_incre_score_last.cpu(), torch.tensor([1],dtype=torch.float)).cuda()
                    tmp_loss[2] = loss_rank

                loss = (tmp_loss * torch.exp(-eta) + eta).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iteration % 20 == 0:
                    print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss, end="")
                    print(' ')
                iteration+= 1
            continue
        elif mode == 'MTL-VF':
            if finetune_backbone:
                pred_final_score, last_output = model_lstm(model_finetune(clip_feats))
            else:
                pred_final_score, last_output = model_lstm(clip_feats)


        if with_skill_classification:
            pred_sp_skill = model_classifier(last_output)

        loss_final_score = (criterion_final_score(pred_final_score, true_final_score)
                            + penalty_final_score(pred_final_score, true_final_score))

        tmp_loss = torch.zeros(3)
        tmp_loss[0] = loss_final_score

        if with_gesture_recognition:
            batch_input_tensor = torch.zeros(1, np.shape(clip_feats_long)[0], len(true_seg.squeeze(0)),
                                             dtype=torch.float)
            batch_input_tensor[0, :, :np.shape(clip_feats_long)[1]] = torch.from_numpy(clip_feats_long)
            batch_input_tensor = batch_input_tensor.cuda()
            if finetune_backbone: batch_input_tensor = model_finetune(batch_input_tensor)
            pred_seg = model_seg(batch_input_tensor, true_seg_mask)
            loss_seg = 0.0
            for p in pred_seg:
                loss_seg += criterion_seg_ce(p.transpose(2, 1).contiguous().view(-1, num_classes), true_seg.view(-1))
                loss_seg += 0.15 * torch.mean(torch.clamp(criterion_seg_mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,max=16) * true_seg_mask[:, :, 1:])
            tmp_loss[1] = loss_seg

        if with_skill_classification:
            loss_sp_skill = criterion_single_label_classifier(pred_sp_skill, true_sp_skill)
            tmp_loss[2] = loss_sp_skill

        if (with_gesture_recognition or with_skill_classification):
            loss = (tmp_loss * torch.exp(-eta) + eta).sum()
        else:
            loss = tmp_loss[0]


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss, ' FS Loss: ', loss_final_score, end="")
            if with_skill_classification:
                  print(' Cls Loss: ', loss_sp_skill, end="")
            if with_gesture_recognition:
                  print(' Seg Loss: ', loss_seg, end="")
            if with_gesture_recognition and with_skill_classification:
                  print(' MTL Uncertainty Parameter: ', eta, end="")
            print(' ')
        iteration += 1

def test(test_dataloader):
    print('Testing starts ...')
    with torch.no_grad():
        pred_scores = []; true_scores = []
        rho_incre_pearson_list = []
        rho_incre_spearman_list = []
        if with_skill_classification:
            pred_gesimp = []
            true_gesimp = []
            pred_sp_skill = []
            true_sp_skill = []

        model_lstm.eval()

        model_finetune.eval()

        if with_skill_classification:
            model_classifier.eval()
        if with_gesture_recognition:
            model_seg.eval()

        # metrics for segmentation
        edit_list = []
        acc_list = []
        recall_list = []
        f1_10_list = []
        f1_25_list = []
        f1_50_list = []
        for _, data in enumerate(test_dataloader):
            pred_incre_scores = []
            true_incre_scores = []

            print(data['name'][0])
            print('----------------')

            true_scores.extend(data['label_final_score'].data.numpy())

            if with_skill_classification:
                true_gesimp.extend([item.data.numpy().tolist()[0] for item in data['label_gesimp']])
                true_sp_skill.extend(data['label_sp_skill'].data.numpy())
            true_impre = data['label_impre']

            clip_feats = torch.Tensor([]).cuda()

            if feature_extractor == 'C3D':
                clip_feats = torch.load(features_path + data['name'][0] + '.pt')
                clip_feats_long = np.load(features_full_path + data['name'][0] + '.npy')
            elif feature_extractor == 'resnet-101':
                clip_feats = clip_feats_long = torch.load(features_path + data['name'][0] + '.pt')
                clip_feats_long = clip_feats_long.squeeze(0).transpose(0, 1).cpu().numpy()


            # LSTM
            if mode == 'IMTL-AGF':
                pred_incre_diffs = []

                true_seg = data['label_seg'].cuda()
                seg_recognition = true_seg.cpu().numpy()[0].tolist()

                cursor = 0
                cursor_list = []

                for _, item in enumerate(seg_recognition):
                    if _ != len(seg_recognition) - 1 and item != seg_recognition[_ + 1]:
                        cursor_list.append((cursor, _))
                        cursor = _ + 1
                    if _ == len(seg_recognition) - 1:
                        cursor_list.append((cursor, _))


                acc_sum = 0.0
                true_impre_alt = preprocessing.normalize(np.array([true_impre]), norm='l1', axis=1)[0].tolist()

                if len(cursor_list) != len(true_impre_alt):
                    print('error data: ' + data['name'][0])
                    continue

                for _, item in enumerate(cursor_list):
                    acc_sum += true_impre_alt[_]
                    if finetune_backbone:
                        temp_final_score, last_output = model_lstm(model_finetune(clip_feats[:, 0:int(item[1] / 8) + 1, :]))
                    else:
                        temp_final_score, last_output = model_lstm(clip_feats[:, 0:int(item[1]/8) + 1, :])

                    pred_incre_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])
                    true_incre_scores.append((data['label_final_score'][0] * acc_sum).item())

                rho_incre_pearson, _ = stats.pearsonr(pred_incre_scores, true_incre_scores)
                rho_incre_spearman, _ = stats.spearmanr(pred_incre_scores, true_incre_scores)

                rho_incre_pearson_list.append(rho_incre_pearson)
                rho_incre_spearman_list.append(rho_incre_spearman)

                pred_scores.append(pred_incre_scores[-1])

                for i in range(len(pred_incre_scores)):
                    if i == 0: pred_incre_diffs.append(pred_incre_scores[i])
                    else: pred_incre_diffs.append(pred_incre_scores[i] - pred_incre_scores[i-1])

                top1_small_ind = pred_incre_diffs.index(min([item for item in pred_incre_diffs if item >= 0]))

                for i in range(len(pred_incre_diffs)):
                    if pred_incre_diffs[i] < 0 or i == top1_small_ind: pred_incre_diffs[i] = 0
                    else: pred_incre_diffs[i] = 1

                true_impre = [int(item.cpu().numpy().tolist()[0]) for item in true_impre]

                correct = 0
                total = 0

                for i in range(len(pred_incre_diffs)):
                    total += 1
                    if pred_incre_diffs[i] == true_impre[i]:
                        correct += 1

                edit = edit_score(pred_incre_diffs, true_impre)

                acc_list.append(100 * float(correct) / total)
                edit_list.append(1.0 * edit)
                recall_list.append(1.0 * recall_score([1-item for item in true_impre], [1-item for item in pred_incre_diffs]))

                continue

            else:
                if finetune_backbone:
                    temp_final_score, last_output = model_lstm(model_finetune(clip_feats))
                else:
                    temp_final_score, last_output = model_lstm(clip_feats)

            if with_gesture_recognition:
                input_x = torch.tensor(clip_feats_long, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                if finetune_backbone: input_x = model_finetune(input_x)
                seg_predictions = model_seg(input_x, torch.ones(input_x.size(), device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")))
                _, predicted_seg = torch.max(seg_predictions[-1].data, 1)
                predicted_seg = predicted_seg.squeeze()
                recognition = []
                for i in range(len(predicted_seg)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted_seg[i].item())]] * 1))
                file_ptr = open(gt_path + data['name'][0] + '.txt', 'r')
                content = file_ptr.read().split('\n')[:-1]

                # metrics calculation
                correct = 0
                total = 0

                overlap = [.1, .25, .5]
                tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

                for i in range(len(recognition)):
                    total += 1
                    if content[i] == recognition[i]:
                        correct += 1
                edit = edit_score(recognition, content)

                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(recognition, content, overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                for s in range(len(overlap)):
                    precision = tp[s] / float(tp[s] + fp[s])
                    recall = tp[s] / float(tp[s] + fn[s])
                    f1 = 2.0 * (precision * recall) / (precision + recall)
                    f1 = np.nan_to_num(f1) * 100
                    if s == 0: f1_10_list.append(f1)
                    elif s == 1: f1_25_list.append(f1)
                    elif s == 2: f1_50_list.append(f1)

                acc_list.append(100*float(correct)/total)
                edit_list.append(1.0 * edit)

            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])


            if with_skill_classification:
                temp_sp_skill = model_classifier(last_output)
                _, temp_sp_skill = torch.max(temp_sp_skill, 1)
                pred_sp_skill.extend(temp_sp_skill.cpu().numpy())

        if with_skill_classification and mode == 'MTL-VF':
            print('Predicted self-proclaim skill: ', pred_sp_skill)
            print('True self-proclaim skill: ', true_sp_skill)
            total_count = 0
            correct_count = 0
            for sp in range(len(true_sp_skill)):
                if pred_sp_skill[sp] == true_sp_skill[sp]: correct_count += 1
                total_count += 1
            sp_accuracy = float(correct_count) / total_count
            print('Accuracy: ', sp_accuracy)
        else:
            sp_accuracy = 0.0

        if with_gesture_recognition and mode == 'MTL-VF':
            print("Seg Acc: %.4f" % np.mean(acc_list))
            print('Seg Edit: %.4f' % np.mean(edit_list))
            print('Seg F1@10: %.4f' % np.mean(f1_10_list))
            m_acc = np.mean(acc_list)
            m_edit = np.mean(edit_list)
            m_f1_10 = np.mean(f1_10_list)
            m_f1_25 = np.mean(f1_25_list)
            m_f1_50 = np.mean(f1_50_list)
            m_recall = 0.0

        elif mode == 'IMTL-AGF':
            print("Feedback Acc: %.4f" % np.mean(acc_list))
            print('Feedback Edit: %.4f' % np.mean(edit_list))
            print('Feedback Recall: %.4f' % np.mean(recall_list))
            m_acc = np.mean(acc_list)
            m_edit = np.mean(edit_list)
            m_recall = np.mean(recall_list)
            m_f1_10 = 0.0
            m_f1_25 = 0.0
            m_f1_50 = 0.0

        else:
            m_acc = 0.0
            m_edit = 0.0
            m_recall = 0.0
            m_f1_10 = 0.0
            m_f1_25 = 0.0
            m_f1_50 = 0.0

        rho, p = stats.spearmanr(pred_scores, true_scores)

        rho_incre_pearson = np.mean(rho_incre_pearson_list)
        rho_incre_spearman = np.mean(rho_incre_spearman_list)


        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        print('Correlation: ', rho)
        print('Incremental Correlation (Pearson): ', rho_incre_pearson)
        print('Incremental Correlation (Spearman): ', rho_incre_spearman)

        return rho, rho_incre_pearson, rho_incre_spearman, sp_accuracy, m_acc, m_edit, m_recall, m_f1_10, m_f1_25, m_f1_50

def main(set_name, set_no):
    # task uncertainty to balance the losses
    global eta
    eta = nn.Parameter(torch.Tensor([0.0, 0.0, 0.0]))     # -3.0, -1.5, 1.0

    parameters_2_optimize_SA = list(model_lstm.parameters())
    parameters_2_optimize_named = (list(model_lstm.named_parameters()))

    parameters_2_optimize_FC = list(model_finetune.parameters())
    parameters_2_optimize_named = parameters_2_optimize_named + list(model_finetune.named_parameters())

    if with_gesture_recognition:
        parameters_2_optimize_SEG = list(model_seg.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_seg.named_parameters())

    if with_skill_classification:
        parameters_2_optimize_CLS = list(model_classifier.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_classifier.named_parameters())


    if with_gesture_recognition and with_skill_classification and mode == 'MTL-VF':
        optimizer = optim.Adam([{'params': parameters_2_optimize_SA, 'lr': SA_learning_rate},
                                {'params': parameters_2_optimize_FC, 'lr': SA_learning_rate},
                                {'params': parameters_2_optimize_SEG, 'lr': SEG_learning_rate},
                                {'params': parameters_2_optimize_CLS, 'lr': CLS_learning_rate},
                                {'params': eta, 'lr': ETA_learning_rate}])
    elif with_skill_classification:
        optimizer = optim.Adam([{'params': parameters_2_optimize_SA, 'lr': SA_learning_rate},
                                {'params': parameters_2_optimize_FC, 'lr': SA_learning_rate},
                                {'params': parameters_2_optimize_CLS, 'lr': CLS_learning_rate},
                                {'params': eta, 'lr': ETA_learning_rate}])
    else:
        optimizer = optim.Adam([{'params': parameters_2_optimize_SA, 'lr': SA_learning_rate},
                                {'params': parameters_2_optimize_FC, 'lr': SA_learning_rate}])

    criterions = {}
    criterion_final_score = nn.MSELoss()
    penalty_final_score = nn.L1Loss()
    criterion_incremental_subscore = nn.MarginRankingLoss()
    criterions['criterion_final_score'] = criterion_final_score
    criterions['penalty_final_score'] = penalty_final_score
    criterions['criterion_incremental_subscore'] = criterion_incremental_subscore
    if with_skill_classification:
        criterion_multi_label_classifier = nn.BCEWithLogitsLoss()
        criterion_single_label_classifier = nn.CrossEntropyLoss()
        criterions['criterion_multi_label_classifier'] = criterion_multi_label_classifier
        criterions['criterion_single_label_classifier'] = criterion_single_label_classifier
    if with_gesture_recognition:
        criterion_seg_ce = nn.CrossEntropyLoss(ignore_index=-100)
        criterion_seg_mse = nn.MSELoss(reduction='none')
        criterions['criterion_seg_ce'] = criterion_seg_ce
        criterions['criterion_seg_mse'] = criterion_seg_mse


    train_dataset = VideoDataset('train', set_name, set_no)
    test_dataset = VideoDataset('test', set_name, set_no)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print('Training set size: ' + str(len(train_dataloader)) + ' and test set size: ' + str(len(test_dataloader)))

    rho_all = []
    rho_incre_pearson_all = []
    rho_incre_spearman_all = []
    sp_acc_all = []
    acc_all = []
    edit_all = []
    recall_all = []
    f1_10_all = []
    f1_25_all = []
    f1_50_all = []

    for epoch in range(max_epochs):
        print('-------------------------------------------------------------------------------------------------------')
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])

        train(train_dataloader, optimizer, criterions, epoch)
        rho, rho_incre_pearson, rho_incre_spearman, sp_acc, acc, edit, recall, f1_10, f1_25, f1_50 = test(test_dataloader)

        rho_all.append(rho)
        rho_incre_pearson_all.append(rho_incre_pearson)
        rho_incre_spearman_all.append(rho_incre_spearman)
        sp_acc_all.append(sp_acc)
        acc_all.append(acc)
        edit_all.append(edit)
        recall_all.append(recall)
        f1_10_all.append(f1_10)
        f1_25_all.append(f1_25)
        f1_50_all.append(f1_50)

    return rho_all, rho_incre_pearson_all, rho_incre_spearman_all, sp_acc_all, acc_all, edit_all, recall_all, f1_10_all, f1_25_all, f1_50_all


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    rho_loso_list = []
    rho_louo_list = []

    rho_incre_pearson_loso_list = []
    rho_incre_pearson_louo_list = []

    rho_incre_spearman_loso_list = []
    rho_incre_spearman_louo_list = []


    sp_acc_loso_list = []
    sp_acc_louo_list = []

    acc_loso_list = []
    acc_louo_list = []

    edit_loso_list = []
    edit_louo_list = []

    recall_loso_list = []
    recall_louo_list = []

    f1_10_louo_list = []
    f1_25_louo_list = []
    f1_50_louo_list = []

    if mode == 'MTL-VF': print('Running MTL-VF framework...')
    elif mode == 'IMTL-AGF': print('Running IMTL-AGF framework...')
    else:
        print('Only MTL-VF and IMTL-AGF are supported.')
        exit()

    for i in range(1,14): # knot 1,14; needle 1,13
        model_lstm = LSTM_final()
        model_lstm = model_lstm.cuda()
        print('Main Head Enabled.')

        model_finetune = fc_finetune()
        model_finetune.cuda()


        if with_skill_classification:
            model_classifier = skill_classifier()
            model_classifier = model_classifier.cuda()
            print('Skill Classification Head Enabled.')

        if with_gesture_recognition:
            model_seg = MultiStageModel(num_stages, num_layers, num_f_maps, features_dim, num_classes)
            model_seg = model_seg.cuda()
            print('Gesture Recognition Head Enabled.')

        if i <= 5:
            rho_loso_all, rho_incre_pearson_loso_all, rho_incre_spearman_loso_all, sp_acc_loso_all, acc_loso_all, edit_loso_all, recall_loso_all, f1_10_loso_all, f1_25_loso_all, f1_50_loso_all = main('LOSO', str(i))

            rho_loso_list.extend(rho_loso_all)
            rho_incre_pearson_loso_list.extend(rho_incre_pearson_loso_all)
            rho_incre_spearman_loso_list.extend(rho_incre_spearman_loso_all)
            sp_acc_loso_list.extend(sp_acc_loso_all)

            acc_loso_list.extend(acc_loso_all)
            edit_loso_list.extend(edit_loso_all)
            recall_loso_list.extend(recall_loso_all)

        else:
            rho_louo_all, rho_incre_pearson_louo_all, rho_incre_spearman_louo_all, sp_acc_louo_all, acc_louo_all, edit_louo_all, recall_louo_all, f1_10_louo_all, f1_25_louo_all, f1_50_louo_all = main('LOUO', str(i-5))
            rho_louo_list.extend(rho_louo_all)
            rho_incre_pearson_louo_list.extend(rho_incre_pearson_louo_all)
            rho_incre_spearman_louo_list.extend(rho_incre_spearman_louo_all)
            sp_acc_louo_list.extend(sp_acc_louo_all)

            acc_louo_list.extend(acc_louo_all)
            edit_louo_list.extend(edit_louo_all)
            recall_louo_list.extend(recall_louo_all)
            f1_10_louo_list.extend(f1_10_louo_all)
            f1_25_louo_list.extend(f1_25_louo_all)
            f1_50_louo_list.extend(f1_50_louo_all)



