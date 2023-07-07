import numpy as np
import math
import scipy
from skimage.transform import rescale
import pygame
import neural_network

pygame.init()

def draw_pixels_on_mouse_movement(mouse_event, image100x100):
    mouse_pressed = event.dict["buttons"][0]
    if not mouse_pressed: return
    cur_x, cur_y = event.dict["pos"]
    dif_x, dif_y = event.dict["rel"]
    prev_x = cur_x - dif_x
    prev_y = cur_y - dif_y
    mouse_distance = math.sqrt((cur_x - prev_x)**2 + (cur_y - prev_y)**2)
    intermediates = int(mouse_distance // (DRAW_IMAGE_PIXEL_SIZE / 2))
    draw_pixels_on_mouse_click(prev_x, prev_y, image100x100)
    draw_pixels_on_mouse_click(cur_x, cur_y, image100x100)
    
    for intermediate in range(intermediates):
        mouse_x = prev_x + (cur_x - prev_x) * (intermediate / intermediates)
        mouse_y = prev_y + (cur_y - prev_y) * (intermediate / intermediates)
        draw_pixels_on_mouse_click(mouse_x, mouse_y, image100x100)
        

def draw_pixels_on_mouse_click(mouse_x, mouse_y, image100x100):
    pixel_row = int(mouse_y // DRAW_IMAGE_PIXEL_SIZE)
    pixel_column = int(mouse_x // DRAW_IMAGE_PIXEL_SIZE)
    
    if not(0 <= pixel_row < DRAW_IMAGE_SIZE and 0 <= pixel_column < DRAW_IMAGE_SIZE):
        return
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            
            adjacent_row = pixel_row + i
            adjacent_column = pixel_column + j
            
            if not(0 <= adjacent_row < DRAW_IMAGE_SIZE 
                   and 0 <= adjacent_column < DRAW_IMAGE_SIZE):
                continue

            gray_value = 1 - ((abs(i) + abs(j)) / 10)
                
            if image100x100[pixel_row + i, pixel_column + j] < gray_value:
                image100x100[pixel_row + i, pixel_column + j] = gray_value
                
                
def scale_image(image100x100):
    top = DRAW_IMAGE_SIZE-1
    bottom = 0
    left = DRAW_IMAGE_SIZE-1
    right = 0
    for row in range(DRAW_IMAGE_SIZE):
        for column in range(DRAW_IMAGE_SIZE):
            pixel = image100x100[row, column]
            if pixel > 0.01:
                top = min(top, row)
                bottom = max(bottom, row)
                left = min(left, column)
                right = max(right, column)
    width = right - left
    height = bottom - top
    scale_factor = min(NETWORK_DIGIT_SIZE / width, NETWORK_DIGIT_SIZE / height)
    resized_image = rescale(image100x100[top:bottom+1, left:right+1], scale_factor)
    padded_image = np.pad(resized_image, ((NETWORK_IMAGE_BORDER,)*2, (NETWORK_IMAGE_BORDER,)*2))
    return padded_image


def center_image(image28x28):
    center_of_mass = scipy.ndimage.center_of_mass(image28x28)
    image_center = (NETWORK_IMAGE_SIZE - 1) / 2
    shift_x = int(image_center - center_of_mass[0])
    shift_y = int(image_center - center_of_mass[1])
    padded_image = np.pad(image28x28, ((NETWORK_IMAGE_SIZE,)*2, (NETWORK_IMAGE_SIZE,)*2))
    rolled_image = np.roll(padded_image, (shift_x, shift_y), axis=(0,1))
    img_start = NETWORK_IMAGE_SIZE
    img_end = img_start + NETWORK_IMAGE_SIZE
    centered_image = rolled_image[img_start:img_end, img_start:img_end]
    return centered_image


def draw_image(image):
    image_size = len(image[0])
    pixel_size = WINDOW_SIZE / image_size
    for row in range(len(image)):
        for column in range(len(image[0])):
            pixel_value = image[row, column]
            pixel_color = ((1-pixel_value)*255,)*3
            pygame.draw.rect(screen, pixel_color,(column*pixel_size, row*pixel_size,
                                                  pixel_size+1, pixel_size+1))
    
WINDOW_SIZE = pygame.display.Info().current_h / 2
DRAW_IMAGE_SIZE = 100
DRAW_IMAGE_PIXEL_SIZE =  WINDOW_SIZE / DRAW_IMAGE_SIZE
NETWORK_IMAGE_SIZE = 28
NETWORK_DIGIT_SIZE = 20
NETWORK_IMAGE_BORDER = int((NETWORK_IMAGE_SIZE - NETWORK_DIGIT_SIZE) / 2)
FPS = 60

draw_img = np.zeros((DRAW_IMAGE_SIZE, DRAW_IMAGE_SIZE))
network_img = np.zeros((NETWORK_IMAGE_SIZE, NETWORK_IMAGE_SIZE))
draw_mode = True

network = neural_network.get_pretrained_network()

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Neuronal network by Pascal")
run = True
clock = pygame.time.Clock()


while run:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        
        if event.type == pygame.QUIT:
            run = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if draw_mode:
                left_click = event.dict["button"] == 1
                mouse_x, mouse_y = event.dict["pos"]
                if left_click:
                    draw_pixels_on_mouse_click(mouse_x, mouse_y, draw_img)
        
        if event.type == pygame.MOUSEMOTION:
            if draw_mode:
                draw_pixels_on_mouse_movement(event, draw_img)
            
        if event.type == pygame.KEYDOWN:
            if event.dict["key"] == pygame.K_SPACE:
                draw_mode = True
                draw_img = np.zeros((DRAW_IMAGE_SIZE, DRAW_IMAGE_SIZE))
            if event.dict["key"] == pygame.K_RETURN:
                if draw_mode:
                    if np.sum(draw_img) <= 0.01: continue
                    draw_mode = False
                    network_img = scale_image(draw_img)
                    network_img = center_image(network_img)
                    network.print_prediction(network_img.ravel())
                
    screen.fill((255,255,255))
    cur_img = draw_img if draw_mode else network_img
    draw_image(cur_img)
    pygame.display.flip()
        
    
pygame.display.quit()