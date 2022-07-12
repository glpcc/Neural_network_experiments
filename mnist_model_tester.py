import pygame
import pickle
import numpy as np
from src.neural_network import NeuralNetwork

def paint_direction(pixels, indc_x, indc_y, strength, direction: tuple[int, int]):
    if indc_x < 28 - direction[0] and indc_x > -1 - direction[0] and indc_y < 28 - direction[1] and indc_y > -1 - direction[1]:
        if pixels[indc_y + direction[1]][indc_x + direction[0]][2] < strength:
            pixels[indc_y + direction[1]][indc_x + direction[0]][2] = strength


window_height = 840
window_width = 840
canvas = pygame.display.set_mode((window_width,window_height))
pygame.init()
runnig = True
pixels = [[[30*j, 30*i, 0] for j in range(28)] for i in range(28)]

Font=pygame.font.SysFont('timesnewroman',  60)
text = Font.render('',False,(0,0,0))
# Load the network object
f = open('models/mnist_predictor_acc0.932.obj', 'rb')
net: NeuralNetwork = pickle.load(f)
while runnig:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            runnig = False

        # When pressed enter the network would predict the result
        if event.type == pygame.KEYDOWN:
            if  event.key == pygame.K_RETURN:
                img = np.array([[[j[2] for j in i] for i in pixels]])/255
                out = net.feed_forward(img)
                text = Font.render(str(np.argmax(out)), True, (255,255,255))
            if event.key == pygame.K_r:
                for i in pixels:
                    for j in i:
                        j[2] = 0

    canvas.fill((0,0,0))
    for i in pixels:
        for j in i:
            pygame.draw.rect(canvas,(j[2],j[2],j[2]),(j[0],j[1],30,30))

    canvas.blit(text,(30,700))
    # Draw where the mouse is when clicked
    if pygame.mouse.get_pressed()[0]:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        indc_x = mouse_x//30
        indc_y = mouse_y//30
        paint_direction(pixels,indc_x,indc_y,253,(0,0))
        paint_direction(pixels,indc_x,indc_y,100,(0,1))
        paint_direction(pixels,indc_x,indc_y,100,(0,-1))
        paint_direction(pixels,indc_x,indc_y,100,(1,0))
        paint_direction(pixels,indc_x,indc_y,100,(-1,0))
        paint_direction(pixels,indc_x,indc_y,50,(1,1))
        paint_direction(pixels,indc_x,indc_y,50,(-1,-1))
        paint_direction(pixels,indc_x,indc_y,50,(1,-1))
        paint_direction(pixels,indc_x,indc_y,50,(-1,1))
        # paint_direction(pixels,indc_x,indc_y,50,(0,2))
        # paint_direction(pixels,indc_x,indc_y,50,(0,-2))
        # paint_direction(pixels,indc_x,indc_y,50,(2,0))
        # paint_direction(pixels,indc_x,indc_y,50,(-2,0))

    pygame.display.update()
