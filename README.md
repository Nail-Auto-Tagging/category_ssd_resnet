# Crop_and_Classify
cv2.imread()의 출력값을 Crop_image()를 통해 nail을 crop하고, get_representation()을 통해 각 nail의 색을 판단 후 결과값을 list의 형태로 출력한다.

example)

image = cv2.imread(...)

cropped = Crop_image(image)

result = get_representation(resnet50, cropped)

""""""""""""""

result = [1 1 1 1 1 1 1 1]
