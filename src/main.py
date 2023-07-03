import numpy as np
import math
import scipy
import pygame
import neural_network

pygame.init()

def draw_pixels_on_mouse_movement(mouse_event):
    mouse_pressed = event.dict["buttons"][0]
    if not mouse_pressed: return
    cur_x, cur_y = event.dict["pos"]
    dif_x, dif_y = event.dict["rel"]
    prev_x = cur_x - dif_x
    prev_y = cur_y - dif_y
    mouse_distance = math.sqrt((cur_x - prev_x)**2 + (cur_y - prev_y)**2)
    intermediates = int(mouse_distance // (PIXEL_SIZE / 2))
    draw_pixels_on_mouse_click(prev_x, prev_y)
    draw_pixels_on_mouse_click(cur_x, cur_y)
    
    for intermediate in range(intermediates):
        mouse_x = prev_x + (cur_x - prev_x) * (intermediate / intermediates)
        mouse_y = prev_y + (cur_y - prev_y) * (intermediate / intermediates)
        draw_pixels_on_mouse_click(mouse_x, mouse_y)
        

def draw_pixels_on_mouse_click(mouse_x, mouse_y):
    pixel_row = int(mouse_y // PIXEL_SIZE)
    pixel_column = int(mouse_x // PIXEL_SIZE)
    
    if not(0 <= pixel_row < IMAGE_SIZE and 0 <= pixel_column < IMAGE_SIZE):
        return
    
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            
            adjacent_row = pixel_row + i
            adjacent_column = pixel_column + j
            
            if not(0 <= adjacent_row < IMAGE_SIZE 
                   and 0 <= adjacent_column < IMAGE_SIZE):
                continue

            if i == 0 or j == 0:
                gray_value = 1
            else:
                gray_value = 0.5
                
            if image[pixel_row + i, pixel_column + j] < gray_value:
                image[pixel_row + i, pixel_column + j] = gray_value
                

def center_image():
    if np.sum(image) == 0: return image
    center_of_mass = scipy.ndimage.center_of_mass(image)
    shift_x = int(13.5 - center_of_mass[0])
    shift_y = int(13.5 - center_of_mass[1])
    padded_image = np.pad(image, ((30, 30), (30, 30)))
    rolled_image = np.roll(padded_image, (shift_x, shift_y), axis=(0,1))
    centered_image = rolled_image[30:58, 30:58]
    return centered_image


def draw_image():
    for row in range(IMAGE_SIZE):
        for column in range(IMAGE_SIZE):
            pixel_value = image[row, column]
            pixel_color = ((1-pixel_value)*255,)*3
            pygame.draw.rect(screen, pixel_color,(column*PIXEL_SIZE, row*PIXEL_SIZE,
                                                  PIXEL_SIZE+1, PIXEL_SIZE+1))
    
WINDOW_SIZE = pygame.display.Info().current_h / 2
IMAGE_SIZE = 28
PIXEL_SIZE = WINDOW_SIZE / IMAGE_SIZE
FPS = 60

network = neural_network.get_pretrained_network()

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Neuronal network by Pascal")
run = True
clock = pygame.time.Clock()

image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

while run:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        
        if event.type == pygame.QUIT:
            run = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            left_click = event.dict["button"] == 1
            mouse_x, mouse_y = event.dict["pos"]
            if left_click:
                draw_pixels_on_mouse_click(mouse_x, mouse_y)
        
        if event.type == pygame.MOUSEMOTION:
            draw_pixels_on_mouse_movement(event)
            
        if event.type == pygame.KEYDOWN:
            if event.dict["key"] == pygame.K_SPACE:
                image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            if event.dict["key"] == pygame.K_RETURN:
                image = center_image()
                network.print_prediction(image.ravel())
                
    screen.fill((255,255,255))
    draw_image()
    pygame.display.flip()
        
    
pygame.display.quit()