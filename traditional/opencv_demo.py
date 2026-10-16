import cv2


image = cv2.imread("images/paper.jpg")

if image is None:
    raise FileNotFoundError("图片读取失败")


print("原图尺寸：", image.shape)


# 缩放
resized = cv2.resize(image, (300, 300))

print("缩放后尺寸：", resized.shape)


# 裁剪
crop = image[20:100, 30:180]

print("裁剪后尺寸：", crop.shape)


# 保存
cv2.imwrite("results/resized_paper.jpg", resized)
cv2.imwrite("results/crop_paper.jpg", crop)


print("处理完成")