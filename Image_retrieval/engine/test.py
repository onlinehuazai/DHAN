import torch
import numpy as np
import torch.nn.functional as F


def inference(model, Xp_test, yp_test):
    model.eval()
    with torch.no_grad():
        pred_y = model(Xp_test.cuda())
        average_precision_li = []
        for idx in range(len(yp_test)):
            query = pred_y[idx].expand(pred_y.shape)
            label = yp_test[idx]
            sim = F.cosine_similarity(pred_y, query)
            _, indices = torch.topk(sim, 100)
            match_list = yp_test[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []
            for item in match_list[1:]:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1
            if precision_li == []:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)
        mAP = np.mean(average_precision_li)
        return mAP