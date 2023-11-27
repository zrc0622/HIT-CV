import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Stitcher:
    # 拼接函数
    def stitch(self, images, ratio=0.95, reprojThresh=4.0, showMatches=False):
        # 获取输入图片
        (imageA, imageB) = images
        top, bottom, left, right = 400, 400, 400, 400
        imageA = cv2.copyMakeBorder(imageA, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        imageB = cv2.copyMakeBorder(imageB, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        


        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageB.shape[1], imageB.shape[0]))

        # self.cv_show('result', result)
        for i in range(imageA.shape[0]):
            for j in range(imageA.shape[1]):
                if imageA[i, j][0] >= result[i, j][0]:
                    result[i, j] = imageB[i, j]
        # self.cv_show('result', result)
        # 检测是否需要显示图片匹配
        if showMatches:
            (imageA, imageB) = images
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
            (matches, H, status) = M
            vis, ptA, ptB = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis, ptA, ptB)

        # 返回匹配结果
        return result

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.BFMatcher()
        # 检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        best_matches = []
        max_matches_num = 0
        ex_dist = 100
        derta_dist = 10
        for i in range(10):
            matches = []
            dist_list = []
            for m in rawMatches:
                if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                    if ex_dist - derta_dist <= m[0].distance <= ex_dist + derta_dist:
                        matches.append((m[0].trainIdx, m[0].queryIdx))
                        dist_list.append(m[0].distance)
            ex_dist = dist_list[random.randint(0, len(dist_list) - 1)]
            if max_matches_num <= len(matches):
                max_matches_num = len(matches)
                best_matches = matches

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(best_matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in best_matches])
            ptsB = np.float32([kpsB[i] for (i, _) in best_matches])

            print(ptsA.shape)
            
            # 计算变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return matches, H, status
        # 匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        ptAs = []
        ptBs = []
        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
                ptAs.append(ptA)
                ptBs.append(ptB)

        # 返回可视化结果
        return vis, ptAs, ptBs


# 读取拼接图片
imageA = cv2.imread("1.jpg")
imageB = cv2.imread("2.jpg")
imageC = cv2.imread("3.jpg")

# 把图片拼接成全景图
stitcher = Stitcher()
(result_ab, vis_ab, pointab, pointba) = stitcher.stitch([imageA, imageB], showMatches=True)
(result_bc, vis_bc, pointcb, pointbc) = stitcher.stitch([imageC, imageB], showMatches=True)
for point in pointab:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_ab, (x, y), 5, (255, 0, 0), -1)
for point in pointba:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_ab, (x, y), 5, (0, 0, 255), -1)
for point in pointbc:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_bc, (x, y), 5, (255, 0, 0), -1)
for point in pointcb:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_bc, (x, y), 5, (0, 0, 255), -1)
result = result_ab
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result_bc[i, j][0] >= result[i, j][0]:
            result[i, j] = result_bc[i, j]

# plt.figure(figsize=(15, 5))
# plt.subplot(2, 3, 1)
# plt.imshow(vis_ab)
# plt.title("vis_ab", fontsize=12)
# plt.axis("off")
# plt.subplot(2, 3, 2)
# plt.imshow(vis_bc)
# plt.title("vis_bc", fontsize=12)
# plt.axis("off")
# plt.subplot(2, 3, 4)
# plt.imshow(result_ab)
# plt.title("result_ab", fontsize=12)
# plt.axis("off")
# plt.subplot(2, 3, 5)
# plt.imshow(result_bc)
# plt.title("result_bc", fontsize=12)
# plt.axis("off")

# plt.axis("off")
# plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("result", fontsize=12)

plt.show()
