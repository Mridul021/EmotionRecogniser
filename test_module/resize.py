from PIL import Image
for i in range(1, 9):
    image = Image.open('result_images/image{}.png'.format(i))
    new_image = image.resize((600,600))
    new_image.save('result_images/image{}.png'.format(i))

