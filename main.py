import random
import svgwrite
import shutil
import cv2
import tempfile
import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from collections import defaultdict
from zipfile import ZipFile
from os.path import basename
from io import BytesIO


# Function to create a zip file
def zip_files(file_name):
    # create a ZipFile object
    with ZipFile(f"{file_name}.zip", 'w') as zipObj:
        # Iterate over all the files in the directory
        for folderName, subfolders, filenames in os.walk(file_name):
            for filename in filenames:
                # Construct the full filepath of the file
                filePath = os.path.join(folderName, filename)
                # Add file to the ZipFile object
                zipObj.write(filePath, basename(filePath))

# Function to unzip `file_contents`
def unzip_file(file_contents):
    with ZipFile(file_contents) as zip_file:
        # Extract all the files in the zipfile to the current directory
        zip_file.extractall("./")
        # Return the list of all the files in the zipfile
        return zip_file.namelist()

# Function to find the path of the image file
def image_path_find(image_file):
    # Extract the file extension from the image_file 
    file_extension = os.path.splitext(image_file.name)[1]
    # Create a tempoary file with the same extension and save the image there
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(image_file.read())
        return tmp_file.name  # Return the path of the saved image

# Function to remove the extension from `font_file`
def strip_font_extension(font_file):
    return font_file.split(' ')[0]

# Function to sort and append word directions
def sort_word_directions(rur=False, rh=False, ru=False, rv=False, v=False, ud=False, h=False, d=False, snake=False):        
    word_directions = []

    if rur: word_directions.append("rur")
    if rh: word_directions.append("rh")
    if ru: word_directions.append("ru")
    if rv: word_directions.append("rv")
    if v: word_directions.append("v")
    if ud: word_directions.append("ud")
    if h: word_directions.append("h")
    if d: word_directions.append("d")
    if snake: word_directions.append("s")

    # if no word directions are passed, add "v" and "h" by default
    if not word_directions: word_directions.extend(["v", "h"])

    return word_directions

# Function to delete the file or directory at `path`
def delete_files(path):
    # If `path` is a file or a symlink
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # Remove the file or symlink
    elif os.path.isdir(path):  # If `path` is a directory
        shutil.rmtree(path)  # Remove the directory and all its contents

# Function to save rejected words into a text file
def save_rejected_words(rejected_words):
    with open('rejected_words.txt', 'a') as file:
        for word in rejected_words:
            file.write(word + '\n')

    # Add 'rejected_words.txt' to generated files if not already present
    if 'rejected_words.txt' not in generated_files:
        generated_files.append('rejected_words.txt')

# Function to convert hexadecimal color code to RGB color code
def hex_to_rgb(hex_code):
    # Remove any leading '#'
    hex_code = hex_code.lstrip('#')
    # Convert hex to RGB
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# Function to convert a list of words to a comma-separated string
def comma_separated_list(words):  
    return ", ".join(words)  

# Function to estimate the pixel width of text based on font size
def get_text_width(text, font_size):
    return len(text) * font_size

# Function to filter out rejected words from the list of words
def filter_words(words, rejected_words):
    return [word for word in words if word not in rejected_words]

# Function to check if the grid contains any bad words
def check_grid_for_bad_words(grid, bad_words):
    # Check in each row
    for row in grid:
        for word in bad_words:
            if word in ''.join(row):
                return True
    # Check in each column
    for col in range(len(grid[0])):
        for word in bad_words:
            if word in ''.join([grid[row][col] for row in range(len(grid))]):
                return True
    return False

# Function to wrap text into multiple lines based on column width
def wrap_text(text, font_size, column_width):
    lines, line, line_width = [], [], 0
    for word in text.split(' '):
        word_width = len(word) * font_size
        if line_width + word_width <= column_width:
            line.append(word)
            line_width += word_width + font_size
        else:
            lines.append(' '.join(line))
            line, line_width = [word], word_width + font_size
    lines.append(' '.join(line))  # Add the last line
    return lines

# Function to check image file extension, and convert image transparency to white background
def check_and_convert_transparency(file_path):
    extension = '.png'
    if not (hasattr(img_file, 'name') and img_file.name.lower().endswith(extension)) and not file_path.lower().endswith(extension):
        return False

    # Load the image
    image = Image.open(file_path).convert('RGBA')

    # Generate a mask using the alpha channel
    alpha_mask = image.split()[3]

    # Generate background image
    bg = Image.new('RGBA', image.size, (255, 255, 255, 255))

    # Apply mask to the background image
    bg.paste(image, mask=alpha_mask)

    # Save the image in RGB format
    bg.convert('RGB').save(img_file_path, 'JPEG')

    return True

# Function of image-based word search generator
def generate_word_search(words, rows, cols, image_path, threshold, directions, language, bad_words,
                         bad_words_sensitivity, bad_words_toggle):

    # Load an image either with or without transparency
    if check_and_convert_transparency(image_path):
        img = cv2.imread(image_path)
    else:
        img = cv2.imread(image_path)

    # Get image dimensions and find the difference between height and width
    original_height, original_width, _ = img.shape
    is_portrait = original_height > original_width
    difference = abs(original_height - original_width)

    # Add padding to the image to make it square
    if is_portrait:  # This conditional branches handles if original image is portrait
        pad_width = ((0, 0), (difference // 2, difference - difference // 2), (0, 0))
    else:            # This branches handles if original image is landscape
        pad_width = ((difference // 2, difference - difference // 2), (0, 0), (0, 0))
    # padding values are added and the image is updated
    img = np.pad(img, pad_width, mode='constant', constant_values=255)

    # Save padded image 
    cv2.imwrite(image_path, img)

    # Convert the image to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define new dimensions to match the grid size
    target_height, target_width = rows, rows

    # Resize the image to match the grid size
    gray = cv2.resize(gray, (target_width, target_height))

    # Invert the image's colors
    gray = 255 - gray

    # Turning image into binary : all pixels above threshold become white, rest becomes black
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Assign threshold to mask
    mask = thresh

    # Update dimensions to match the resized mask size
    rows, cols = mask.shape

    # Initialize the grid with either empty strings or blanks depending on the mask
    grid = [["" if mask[r][c] else " " for c in range(cols)] for r in range(rows)]

    # Array to store words that could not be placed in the grid
    rejected_words = []

    # Dictionary to store each word's coordinates in the grid
    word_coordinates = {}

    # Define the direction steps for the word placements
    steps = {
        "h": (0, 1),     # Horizontal - left to right
        "v": (1, 0),     # Vertical - top to bottom
        "d": (1, 1),     # Diagonal - top left to bottom right
        "ud": (-1, 1),   # Diagonal - bottom left to top right
        "rh": (0, -1),   # Horizontal - right to left
        "rv": (-1, 0),   # Vertical - bottom to top
        "ru": (-1, -1),  # Diagonal - bottom right to top left
        "rur": (1, -1),  # Diagonal - top right to bottom left
    }

    # Checks if a word can be placed in the grid in a specific direction
    def can_place_word_regular(word, r, c, dir):
        r_step, c_step = steps.get(dir, (None, None))
        if r_step is None or c_step is None:
            raise ValueError("Invalid direction")

        return all(
            0 <= r + i * r_step < rows and
            0 <= c + i * c_step < cols and
            (grid[r + i * r_step][c + i * c_step] == "" or
             grid[r + i * r_step][c + i * c_step] == word[i])
            for i in range(len(word))
        )

    # Recursive path generating function for snaking words
    def generate_snaking_path(remaining_word, r, c, used_cells): 

        if not remaining_word:
            return []

        # Try all directions for the next character
        for dr, dc in steps.values():
            new_r, new_c = r + dr, c + dc
            if (
                0 <= new_r < rows
                and 0 <= new_c < cols
                and (
                    (grid[new_r][new_c] == "" or grid[new_r][new_c] == remaining_word[0])
                    and (new_r, new_c) not in used_cells
                )
            ):
                grid[new_r][new_c] = remaining_word[0]
                used_cells.add((new_r, new_c))
                subpath = generate_snaking_path(
                    remaining_word[1:], new_r, new_c, used_cells
                )
                if subpath is not None:
                    return [(new_r, new_c)] + subpath
                grid[new_r][new_c] = ""
                used_cells.remove((new_r, new_c))

        return None

    # Function to remove invalid characters in a word
    def remove_invalid_chars(word):
        return word.replace(" ", "").replace("-", "").replace("'", "")

    # Iterating over each word to place it in the grid
    for word in words:

        # Store the original word
        original_word = word  

        # Skip the word if it's too long for the grid
        if len(word) > rows and len(word) > cols:
            #st.write(f"Skipping word '{original_word}', as it's too long for the grid.")
            rejected_words.append(original_word)
            continue

        # Remove invalid characters from the word
        word = remove_invalid_chars(word)
        tries_left = 1000
        placed = False

        # Try to place the word in the grid
        while not placed and tries_left > 0:
            # Select a random direction
            dir = random.choice(directions)

            # If we are to place word in snaking pattern
            if dir == 's':
                r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
                path = generate_snaking_path(word, r, c, set())
                if path:
                    for position, char in zip(path, word):
                        r, c = position
                        grid[r][c] = char
                        if word not in word_coordinates:
                            word_coordinates[word] = []
                        word_coordinates[word].append((r, c))
                    placed = True
            # If we are to place word in simple pattern (horizontally, vertically, or diagonally)
            else:
                if dir in ["h", "rh"]:
                    r, c = random.randint(0, min(rows - 1, max(rows - len(word), 0))), random.randint(0, min(cols - 1, max(cols - len(word), 0)))
                elif dir in ["v", "rv"]:
                    r, c = random.randint(0, min(rows - 1, max(rows - len(word), 0))), random.randint(0, min(cols - 1, max(cols - 1, 0)))
                elif dir in ["d", "rur"]:
                    r, c = random.randint(0, min(rows - 1, max(rows - len(word), 0))), random.randint(0, min(cols - 1, max(cols - len(word), 0)))
                elif dir in ["ud", "ru"]:
                    r, c = random.randint(0, min(rows - 1, max(rows - len(word), 0))), random.randint(0, min(cols - 1, max(cols - len(word), 0)))
                else:
                    raise ValueError("Invalid direction")

                # If the word can be placed in the selected direction, add it to the grid and the word_coordinates dictionary
                if can_place_word_regular(word, r, c, dir):
                    r_step, c_step = steps[dir]
                    for i in range(len(word)):
                        grid[r + i * r_step][c + i * c_step] = word[i]
                        if word not in word_coordinates:
                            word_coordinates[word] = []
                        word_coordinates[word].append((r + i * r_step, c + i * c_step))
                    placed = True
                else:
                    tries_left -= 1

            if not placed:
                tries_left -= 1

        # If the word couldn't be placed, add it to the rejected words list
        if not placed:
            rejected_words.append(original_word)

    # If a cell in the grid contains an empty string, fill it with a random letter based on the language
    def fill_random_letters(grid):

        if language == "English":
            letters = "abcdefghijklmnopqrstuvwxyz"
        elif language == "Italian":
            letters = "abcdefghilmnopqrstuvz"
        elif language == "German":
            letters = "abcdefghijklmnopqrstuvwxyzäöüß"
        elif language == "Greek":
            letters = "αβγδεζηθικλμνξοπρσςτυφχψω"
        elif language == "Spanish":
            letters = "ñabcdefghijklmnopqrstuvwxyz"
        elif language == "Numbers":
            letters = "0123456789"

        for r, row in enumerate(grid):
            for c, char in enumerate(row):
                if char == "":
                    grid[r][c] = random.choice(letters)

        return grid

    # Flag to indicate if the generated grid is valid
    grid_is_valid = False  
    # Counter to track the number of attempts to generate a valid grid
    max_attempts = 0  

    # Main loop to generate a grid until it's valid or the maximum attempts are reached
    while not grid_is_valid:
        # Fill the grid with random letters
        grid = fill_random_letters(grid)  
        # Check if bad words filtering is enabled
        if bad_words_toggle == "On":  
            # Check if the grid contains any bad words
            if not check_grid_for_bad_words(grid, bad_words):
                # Grid is valid as it doesn't contain bad words
                grid_is_valid = True  
            else:
                # Increment the attempt counter as bad words are found
                max_attempts += 1  
            if max_attempts > bad_words_sensitivity:
                # Break the loop if maximum attempts to filter bad words are reached
                break  
        else:
            # Grid is valid as bad word filtering is disabled
            grid_is_valid = True  

    # Check if the final grid is valid and process the results accordingly
    if grid_is_valid: 
        # If there are rejected words from the bad word filter
        if rejected_words:  
             # Save the rejected words
            save_rejected_words(rejected_words) 
        # Return the generated grid and associated data
        return grid, rejected_words, word_coordinates  
    
    # Grid is not valid due to the profanity filter reaching maximum attempts
    else:  
        st.error("Profanity filter reached maximum attempts to generate grid")
        # Return None for the grid and the rest of the data
        return None, rejected_words, word_coordinates  

def is_snaking_path(word_positions):
    if len(word_positions) < 2:
        # If there are less than 2 word positions, it's not a snaking path
        return False  

    # Define a helper function to check if two cells are adjacent - horizontally, vertically, or diagonally
    def is_adjacent(cell1, cell2):
        dr, dc = abs(cell1[0] - cell2[0]), abs(cell1[1] - cell2[1])
        return (dr == 0 and dc == 1) or (dr == 1 and dc == 0) or (dr == 1 and dc == 1)
    
    # Loop through the word positions to check for adjacency between consecutive cells
    for i in range(len(word_positions) - 1):
        # If any consecutive cells are not adjacent, it's not a snaking path
        if not is_adjacent(word_positions[i], word_positions[i + 1]):
            return False  
    # If all word positions are adjacent, it's a snaking path
    return True  

# For .png & .jpg
def grid_to_image(*kwargs):
    
    # Function to draw a border around contours for image mask.
    def draw_mask_border(resized_image, position, draw, img_width, img_height, padding, offset_y_grid):
        # Convert the resized image to grayscale format.
        gray_image = resized_image.convert('L')

        # Convert the PIL image to OpenCV format.
        image = np.array(gray_image)

        # Apply threshold to create a binary image and remove noise using morphological operations.
        _, thresh = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours in the binary image.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set the thickness of the contour lines.
        thickness = 2

        # Helper function to draw each individual contour.
        def draw_contour(contour, draw, offset_x, offset_y):
            # Iterate through each point in the contour and draw lines between consecutive points.
            for i in range(len(contour)):
                start = tuple(contour[i][0])
                end = tuple(contour[(i + 1) % len(contour)][0])
                start = (start[0] + offset_x, start[1] + offset_y)
                end = (end[0] + offset_x, end[1] + offset_y)
                draw.line([start, end], fill=(mask_border_opacity,) * 3, width=thickness)

        # Iterate through each contour and draw the borders if they have sufficient area.
        for contour in contours:
            if cv2.contourArea(contour) >= padding:
                draw_contour(contour, draw, position[0], position[1])

    def draw_image(title_pos, words_in_grid=None, highlight_words=False, show_word_list=True):
        # Create a new image with a white background
        img = Image.new("RGB", (img_width, img_height), color="white")
        
        # Create a drawing object for the image
        draw = ImageDraw.Draw(img)
        
        # Calculate the vertical padding based on title position and resolution
        vertical_padding = padding_title * resolution + 85 * resolution

        # Define the starting position for the grid on the image
        grid_start_y = 100 * resolution + padding

        def draw_resized_image(img_file_path, img_width, img_height, grid_start_y, padding, cell_size, shift_by_cells):
            # Read the original image from the provided file path
            original_image = Image.open(img_file_path)

            # Resize the image to fit the grid while maintaining aspect ratio
            image_width, image_height = original_image.size
            aspect_ratio = float(image_width) / float(image_height)
            new_width = img_width - 2 * padding
            new_height = int(new_width / aspect_ratio)

            # Adjust the image height if it exceeds the available space after considering padding and vertical title position
            if new_height > (img_height - vertical_padding - padding):
                new_height = img_height - vertical_padding - padding
                new_width = int(new_height * aspect_ratio)

            # Perform the resizing with anti-aliasing for smoothness
            resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)

            # Calculate the position offset for the resized image
            offset_x = (img_width - new_width) // 2
            offset_y = grid_start_y

            # Return the resized image and its position on the grid
            return resized_image, (offset_x, offset_y)
        
        # Call the 'draw_resized_image' function with the specified parameters
        resized_image, position = draw_resized_image(img_file_path, img_width, img_height, grid_start_y, padding, cell_size, 4.35)

        # Check if 'show_image' is True and 'img_file_path' is not None to proceed with displaying an image.
        if show_image and img_file_path is not None:

            # Convert the image to RGBA mode to support alpha channel.
            resized_image = resized_image.convert("RGBA")

            # If 'grayscale_toggle' is True, convert the image to grayscale.
            if grayscale_toggle:
                # Convert the image to grayscale using ImageOps.
                gray_image = ImageOps.grayscale(resized_image)

                # Merge the grayscale image with the alpha channel from the original image to maintain transparency.
                resized_image = Image.merge('RGBA', (gray_image, gray_image, gray_image, resized_image.split()[3]))

            # Create an alpha layer with the desired opacity level.
            alpha = Image.new("L", resized_image.size, image_opacity)
            resized_image.putalpha(alpha)

            # Adjust the y position for the image to align with the letter grid.
            position_image_y = position[1]
            position = (position[0], position_image_y)

            # Paste the resized image onto the main image 'img' at the specified position.
            img.paste(resized_image, position, resized_image)

            # Calculate the vertical offset for the grid after adding the image.
            offset_y_grid = position[1] + resized_image.size[1]
        else:
            # If 'show_image' is False or 'img_file_path' is None, set the vertical offset to the padding value.
            offset_y_grid = vertical_padding

        # If 'mask_border' is True, draw a border around the inserted image area on the main image 'img'.
        if mask_border:
            position_mask = (position[0], grid_start_y)
            draw_mask_border(resized_image, position_mask, draw, img_width, img_height, padding, offset_y_grid)

        # If 'title_switch' is False, draw the section title on the main image 'img'.
        if not title_switch:
            # Calculate the width and height of the section title text using 'title_font'.
            title_width, title_height = draw.textsize(section, font=title_font)

            # Calculate the x coordinate to center the title on the image.
            title_x = (img_width - title_width) // 2

            # Calculate the y coordinate for the title position.
            title_p = 10 * resolution + padding_title + title_pos

            # Draw the section title on the image.
            draw.text((title_x, title_p), section, font=title_font, fill=(0, 0, 0))

        # Draw the letter grid on the main image 'img'.
        for r, row in enumerate(grid):
            for c, char in enumerate(row):

                # Convert the character to uppercase if 'make_uppercase' is True, else convert to lowercase.
                char = char.upper() if make_uppercase else char.lower()

                # Calculate the width and height of the character using 'font'.
                char_width, char_height = draw.textsize(char, font=font)

                # Calculate the center x and y coordinates for the character position within the grid cell.
                char_center_x = (c * cell_size) + (cell_size - char_width) // 2 + padding
                char_center_y = (r * cell_size) + 100 * resolution + (cell_size - char_height) // 2 + padding

                # Check if the current character position should be highlighted based on 'highlight_words'.
                should_highlight_char = False
                if highlight_words:
                    for word_positions in word_coordinates.values():
                        if (r, c) in word_positions:
                            should_highlight_char = True
                            break

                if not should_highlight_char:
                    # Draw the character on the image at the calculated position.
                    draw.text((char_center_x, char_center_y), char, font=font, fill=(0, 0, 0))

        # Check if words are to be highlighted
        if highlight_words:
            # Iterate over the positions of the words
            for word_positions in word_coordinates.values():  

                # If the chosen style is to outline words
                if highlight_style == 'Outline words':

                    # Get the minimum and maximum rows and columns of the word position
                    min_r = min(word_positions, key=lambda x: x[0])[0]
                    min_c = min(word_positions, key=lambda x: x[1])[1]
                    max_r = max(word_positions, key=lambda x: x[0])[0]
                    max_c = max(word_positions, key=lambda x: x[1])[1]

                    # Determine the orientation of the words
                    is_horizontal = min_r == max_r
                    is_vertical = min_c == max_c
                    is_diagonal = not (is_horizontal or is_vertical)

                    # If the word is diagonal
                    if is_diagonal:
                        # Determine the direction of the diagonal word (bottom left to top right or top left to bottom right)
                        is_bottom_left_to_top_right = all((r - min_r == c - min_c) for r, c in word_positions)
                        is_top_left_to_bottom_right = all((max_r - r == c - min_c) for r, c in word_positions)

                        # Settings for the rectangle attributes and to check the direction 
                        offset = cell_size // 2
                        extension_start = 0  
                        extension_end = 0.5  
                        vertical_offset = 40 * resolution

                        # Draw the outline for the diagonal word in the bottom left to top right direction
                        if is_bottom_left_to_top_right:
                            points = [
                                ((min_c - extension_start) * cell_size + padding + offset, (min_r - extension_start) * cell_size + 100 * resolution + padding),
                                ((max_c + extension_end) * cell_size + padding + cell_size - offset, (max_r + extension_end) * cell_size + 100 * resolution + padding),
                                ((max_c + extension_end) * cell_size + padding, (max_r + extension_end + 1) * cell_size + 100 * resolution + padding - offset),
                                ((min_c - extension_start) * cell_size + padding, (min_r - extension_start + 1) * cell_size + 100 * resolution + padding - offset),
                                ((min_c - extension_start) * cell_size + padding + offset, (min_r - extension_start) * cell_size + 100 * resolution + padding)
                            ]
                            draw.line(points, width=border_width, fill=colour, joint="round")

                        # Draw the outline for the diagonal word in the top left to bottom right direction
                        if is_top_left_to_bottom_right:
                            points = [
                                ((min_c - extension_start) * cell_size + padding + offset, (max_r + extension_start) * cell_size + 100 * resolution + padding + vertical_offset),
                                ((max_c + extension_end) * cell_size + padding + cell_size - offset, (min_r - extension_end) * cell_size + 100 * resolution + padding + vertical_offset),
                                ((max_c + extension_end) * cell_size + padding, (min_r - extension_end) * cell_size + 100 * resolution + padding - offset + vertical_offset),
                                ((min_c - extension_start) * cell_size + padding, (max_r + extension_start) * cell_size + 100 * resolution + padding - offset + vertical_offset),
                                ((min_c - extension_start) * cell_size + padding + offset, (max_r + extension_start) * cell_size + 100 * resolution + padding + vertical_offset)
                            ]
                            draw.line(points, width=border_width, fill=colour, joint="round")

                    # If the word is not diagonal, draw a simple rectangle for the outline
                    else:
                        rect_radius = int(cell_size * 0.05)
                        rect_x = min_c * cell_size + padding
                        rect_y = min_r * cell_size + 100 * resolution + padding
                        rect_width = (max_c - min_c + 1) * cell_size
                        rect_height = (max_r - min_r + 1) * cell_size
                        draw.rounded_rectangle([rect_x, rect_y, rect_x + rect_width, rect_y + rect_height], 
                                            radius=rect_radius, outline=colour, width=border_width)

                # If the chosen style is to strikethrough words
                elif highlight_style == 'Strikethrough words':

                    is_snaking = is_snaking_path(word_positions)

                    if not is_snaking:
                        min_r = min(word_positions, key=lambda x: x[0])[0]
                        min_c = min(word_positions, key=lambda x: x[1])[1]
                        max_r = max(word_positions, key=lambda x: x[0])[0]
                        max_c = max(word_positions, key=lambda x: x[1])[1]

                        midpoint = cell_size // 2

                        # Set the points for horizontal, vertical or diagonal words for drawing strikethrough lines
                        # Horizontal word
                        if min_r == max_r:  
                            points = [
                                (min_c * cell_size + padding, min_r * cell_size + 100 * resolution + padding + midpoint),
                                ((max_c + 1) * cell_size + padding, min_r * cell_size + 100 * resolution + padding + midpoint)
                            ]
                        # Vertical word
                        elif min_c == max_c:  
                            points = [
                                (min_c * cell_size + padding + midpoint, min_r * cell_size + 100 * resolution + padding),
                                (max_c * cell_size + padding + midpoint, (max_r + 1) * cell_size + 100 * resolution + padding)
                            ]
                        # Diagonal word
                        else:  
                            is_diagonal_down_right = all((r - min_r) == (c - min_c) for r, c in word_positions)
                            # Diagonal from top left to bottom right
                            if is_diagonal_down_right:  
                                points = [
                                    (min_c * cell_size + padding, min_r * cell_size + 100 * resolution + padding),
                                    ((max_c + 1) * cell_size + padding, (max_r + 1) * cell_size + 100 * resolution + padding)
                                ]
                            # Diagonal from bottom left to top right
                            else:  
                                points = [
                                    (min_c * cell_size + padding, (max_r + 1) * cell_size + 100 * resolution + padding),
                                    ((max_c + 1) * cell_size + padding, min_r * cell_size + 100 * resolution + padding),
                                ]

                        # Draw the words and the strikethrough line
                        for r, c in word_positions:
                            char = grid[r][c]
                            final_char = char.upper() if make_uppercase else char.lower()

                            char_width, char_height = draw.textsize(final_char, font=font)

                            char_center_x = (c * cell_size) + (cell_size - char_width) // 2 + padding
                            char_center_y = (r * cell_size) + 100 * resolution + (cell_size - char_height) // 2 + padding

                            draw.text((char_center_x, char_center_y), final_char, font=font, fill=(0, 0, 0))

                        draw.line(points, fill=colour, width=border_width)

                    # If the word is snaking, draw a line connecting each cell
                    else:
                        for i, (r, c) in enumerate(word_positions):
                            char = grid[r][c]
                            final_char = char.upper() if make_uppercase else char.lower()

                            char_width, char_height = draw.textsize(final_char, font=font)

                            char_center_x = (c * cell_size) + (cell_size - char_width) // 2 + padding
                            char_center_y = (r * cell_size) + 100 * resolution + (cell_size - char_height) // 2 + padding

                            draw.text((char_center_x, char_center_y), final_char, font=font, fill=(0, 0, 0))

                            if i < len(word_positions) - 1:
                                # Draw a line connecting the center of the current cell to the center of the next cell
                                next_r, next_c = word_positions[i + 1]
                                point1 = (c * cell_size + 0.5 * cell_size + padding, r * cell_size + 100 * resolution + 0.5 * cell_size + padding)
                                point2 = (next_c * cell_size + 0.5 * cell_size + padding, next_r * cell_size + 100 * resolution + 0.5 * cell_size + padding)
                                draw.line([point1, point2], fill=colour, width=border_width)

                # If the chosen style is neither outline or strikethrough, then just draw the words 
                else:
                    for r, c in word_positions:
                        char = grid[r][c]
                        final_char = char.upper() if make_uppercase else char.lower()

                        char_width, char_height = draw.textsize(final_char, font=font)

                        char_center_x = (c * cell_size) + (cell_size - char_width) // 2 + padding
                        char_center_y = (r * cell_size) + 100 * resolution + (cell_size - char_height) // 2 + padding

                        draw.text((char_center_x, char_center_y), final_char, font=font, fill=colour)

        if highlight_style == 'Outline words' and highlight_words:
    
            # Iterate through the word_positions dictionary containing lists of (row, column) coordinates for each word to highlight.
            for word_positions in word_coordinates.values():
                
                # For each position of a word, extract the row (r) and column (c) coordinates.
                for r, c in word_positions:

                    # Determine whether to use uppercase or lowercase for the character at the current position.
                    char = grid[r][c].upper() if make_uppercase else grid[r][c].lower()

                    # Get the width and height of the character using the specified font.
                    char_width, char_height = draw.textsize(char, font=font)

                    # Calculate the center coordinates (char_center_x, char_center_y) to draw the character in the cell.
                    char_center_x = (c * cell_size) + (cell_size - char_width) // 2 + padding
                    char_center_y = (r * cell_size) + 100 * resolution + (cell_size - char_height) // 2 + padding

                    # Draw the character on the grid at the calculated center coordinates using the provided font and fill color (black).
                    draw.text((char_center_x, char_center_y), char, font=font, fill=(0, 0, 0))

        # Check if gridlines or border should be drawn
        if gridlines or border:
            # Loop through rows (r) of the grid and draw horizontal lines
            for r in range(len(grid) + 1):
                # If gridlines are turned off and this is not the first or last row, skip drawing the line
                if not gridlines and r != 0 and r != len(grid):
                    continue
                # Draw a horizontal line at the current row
                draw.line([padding, 100 * resolution + padding + r * cell_size, padding + len(grid[0]) * cell_size, 100 * resolution + padding + r * cell_size], fill="gray")

            # Loop through columns (c) of the grid and draw vertical lines
            for c in range(len(grid[0]) + 1):
                # If gridlines are turned off and this is not the first or last column, skip drawing the line
                if not gridlines and c != 0 and c != len(grid[0]):
                    continue
                # Draw a vertical line at the current column
                draw.line([padding + c * cell_size, 100 * resolution + padding, padding + c * cell_size, 100 * resolution + padding + len(grid) * cell_size], fill="gray")
    
        def draw_column(draw, img, img_width, img_height, words_section_height, words_start_y, final_words, words_font, grid_padding, column_count):
            # Find the widest word for each column
            # Initialize a list to store the widest word width for each column
            widest_word_widths = [0] * column_count  
            # Loop through each word in the list
            for i, word in enumerate(final_words):  
                # Calculate the column index for the current word
                column_index = i % column_count  
                # Get the width of the current word using the specified font
                word_width = draw.textsize(word, font=words_font)[0]  
                # Check if the current word is wider than the widest word in the column
                if word_width > widest_word_widths[column_index]:  
                    # Update the widest word width for the column if needed
                    widest_word_widths[column_index] = word_width  

            # Calculate the total width required, including padding between columns
            # Set the padding between columns to 20 pixels
            padding_between_columns = 20  
            # Calculate the total width needed for all columns
            total_required_width = sum(widest_word_widths) + padding_between_columns * (column_count - 1)  

            # Calculate the starting x-coordinate for each column
            # Initialize a list to store the starting x-coordinate for each column
            columns_start_x = [0] * column_count  

            # Calculate the center padding
            # Calculate the padding needed to center the columns within the image
            center_padding = (img_width - total_required_width) // 2  
            # Check if the center padding is less than the specified grid padding
            if center_padding < grid_padding:  
                # Adjust image width and recalculate center_padding
                # Store the original image width
                old_img_width = img_width  
                # Set the new image width to accommodate the columns and additional grid padding
                img_width = total_required_width + grid_padding * 2  
                # Create a new resized image with the updated width and original height
                img_resized = Image.new("RGB", (img_width, img_height), color="white")
                # Paste the original image into the center of the new image  
                img_resized.paste(img, (img_width // 2 - old_img_width // 2, 0))  
                # Update the image variable with the resized image
                img = img_resized  
                # Create a new drawing object for the updated image
                draw = ImageDraw.Draw(img)  
                # Recalculate the center padding
                center_padding = (img_width - total_required_width) // 2  

            # Loop through each column
            for i in range(column_count):  
                if i == 0:
                    # The first column starts at the calculated center padding
                    columns_start_x[i] = center_padding  
                else:
                    # Calculate the starting x-coordinate for the next column
                    columns_start_x[i] = columns_start_x[i - 1] + widest_word_widths[i - 1] + padding_between_columns  
            # Initialize the y-coordinate for drawing the word list in columns
            current_y = [words_start_y] * column_count  # Create a list to store the current y-coordinate for each column

            # Iterate over the words list and draw the words in columns
            # Loop through each word in the final list
            for i, word in enumerate(final_words):  
                # Get the width and height of the current word using the specified font
                word_width, word_height = draw.textsize(word, font=words_font)  
                # Calculate the column index for the current word
                column_index = i % column_count  
                # Calculate the x-coordinate offset to center the word in the column
                text_offset_x = (widest_word_widths[column_index] - word_width) // 2  
                # Draw the word in the current column at the appropriate position
                draw.text((columns_start_x[column_index] + text_offset_x, current_y[column_index]), word, font=words_font, fill=(0, 0, 0))  
                # Update the current y-coordinate for the column after drawing the word
                current_y[column_index] += word_height + grid_padding  

            # Find the maximum y-coordinate among all columns
            max_current_y = max(current_y)  
            # Check if the total height required for all columns exceeds the original image height
            if max_current_y > img_height:  
                # Create a new resized image with enough height to accommodate all columns and additional grid padding
                img_resized = Image.new("RGB", (img_width, max_current_y + grid_padding), color="white")  
                # Paste the original image into the new image
                img_resized.paste(img)  
                # Update the image variable with the resized image
                img = img_resized  

            # Return the required width as well as the updated image height and width
            # Calculate the required width for the columns
            required_width = columns_start_x[-1] + widest_word_widths[-1] + grid_padding  
            # Return the updated image, total height, and required width
            return img, max_current_y + grid_padding, required_width  

        def calculate_max_line_height(lines, words_font):
            # Initialize a variable 'max_line_height' to -1. This variable will store the maximum line height found.
            max_line_height = -1
            
            # Iterate through each line in the 'lines' list.
            for line in lines:
                # Use the 'draw.textsize()' function to get the width and height of the current 'line' of text
                # The width is ignored (hence '_', a throwaway variable), and only the height ('line_height') is kept.
                _, line_height = draw.textsize(line, font=words_font)
                
                # Update 'max_line_height' to be the maximum value between its current value and the 'line_height'.
                max_line_height = max(max_line_height, line_height)
            
            # After processing all lines, return the maximum line height found in the 'lines' list.
            return max_line_height
        
        # Define a dictionary called 'column_count_map' that maps column descriptions to their respective counts.
        column_count_map = {
            "Single column": 1,
            "Double column": 2,
            "Triple column": 3,
            "Quad column": 4
        }

        # Draw the words if word_list_switch is True
        if not word_list_switch and show_word_list:
            # Determine the number of columns to display based on the word_list_format
            column_count = column_count_map.get(word_list_format, 2)

            # Filter out any rejected words from the list of words
            final_words = filter_words(words, rejected_words)

            # Calculate the width of each column based on the image width and padding
            column_width = (img_width - grid_padding * 3) // 2

            # Calculate the starting Y position for drawing the words section, including word_list_pos
            words_start_y = img_height - words_section_height - grid_padding * 8 + padding_word_list + word_list_pos

            if word_list_format == "Comma separated list":
                # Generate a comma-separated list of words
                comma_separated_words = comma_separated_list(final_words)

                # Initialize variables for word wrapping
                line = ''
                lines = []
                max_line_width = img_width - 2 * grid_padding - 150

                # Iterate through each word in the comma-separated list
                for word in comma_separated_words.split(', '):

                    # Check if this is the last word in the list
                    is_last_word = (word == comma_separated_words.split(', ')[-1])

                    # Create a new line by adding the current word and a comma (if not the last word)
                    if not is_last_word:
                        new_line = line + word + ', '
                    else:
                        new_line = line + word

                    # Calculate the width of the new line when drawn with the specified font
                    new_line_width, _ = draw.textsize(new_line, font=words_font)

                    # Check if the new line fits within the maximum line width
                    if new_line_width <= max_line_width:
                        # If it fits, update the line with the new content
                        line = new_line
                    else:
                        # If it doesn't fit, add the current line to the list of lines and start a new line
                        lines.append(line.strip())
                        if not is_last_word:
                            line = word + ', '
                        else:
                            line = word

                # Add the last line to the list of lines (remove trailing comma if present)
                lines.append(line.rstrip(', '))

                # Calculate the maximum line height for all the lines using the words_font
                max_line_height = calculate_max_line_height(lines, words_font)

                # Calculate the required height and width of the image to accommodate all the lines
                required_height = words_start_y + len(lines) * (max_line_height + grid_padding)
                required_width = img_width

            else:
                # For other word_list_formats, draw a dummy column to calculate the required width
                _, required_column_height, required_column_width = draw_column(draw, img, img_width, img_height, words_section_height, words_start_y, final_words, words_font, grid_padding, column_count)

                # Determine the final required height and width of the image
                required_height = max(img_height, required_column_height)
                required_width = max(img_width, required_column_width)

            # Check if image size needs to be adjusted
            if img_width != required_width or img_height != required_height:
                # Calculate the new dimensions for the resized image
                img_resized_width = max(img_width, required_width)  # Take the maximum required width for both the comma-separated and column-based word lists
                img_resized_height = max(img_height, required_height)

                # Create a new blank white image with the resized dimensions
                img_resized = Image.new("RGB", (img_resized_width, img_resized_height), color="white")

                # Paste the original image onto the new resized image
                img_resized.paste(img)

                # Update the image reference and the drawing object to the resized versions
                img = img_resized
                draw = ImageDraw.Draw(img)

            # Now actually draw the comma-separated list or columns
            if word_list_format == "Comma separated list":
                # Initialize starting Y coordinate for drawing lines of words
                y = words_start_y

                # Initialize the maximum line height to keep track of the tallest line
                max_line_height = -1

                # Iterate through each line in the list of lines
                for line in lines:
                    # Calculate the width and height of the line using the words_font
                    line_width, line_height = draw.textsize(line, font=words_font)

                    # Calculate the horizontal offset to center the line within the image width
                    text_offset_x = (img_width - line_width) // 2

                    # Draw the line of words on the image at the current Y coordinate
                    draw.text((text_offset_x, y), line, font=words_font, fill=(0, 0, 0))

                    # Update the maximum line height if the current line is taller than the previous maximum
                    max_line_height = max(max_line_height, line_height)

                    # Move the Y coordinate down to the next position for the next line
                    y += max_line_height + grid_padding

            else:
                # Draw columns for non-comma-separated word lists
                img, _, _ = draw_column(draw, img, img_width, img_height, words_section_height, words_start_y, final_words, words_font, grid_padding, column_count)
        
        # Return the image
        return img

    # Remove the file extension from the font_choice variable
    font_edit = strip_font_extension(font_choice)

    # Calculate the padding based on the resolution
    padding = 50 * resolution

    # Load three different fonts for different purposes
    title_font = ImageFont.truetype(font_edit + ".ttf", header_size * resolution)  # Font for the title/header
    font = ImageFont.truetype(font_edit + ".ttf", letter_size * resolution)  # Font for letters in the grid
    words_font = ImageFont.truetype(font_edit + ".ttf", wordlist_size * resolution)  # Font for words in the word list

    # Calculate the cell size and grid_padding for drawing the grid
    cell_size = max(letter_size, 40) * resolution  # Ensure cell size is large enough to accommodate varying font sizes
    grid_padding = cell_size // 4

    # Calculate the total image width based on the grid size and padding
    img_width = len(grid[0]) * cell_size + 2 * padding

    # Calculate the height of the words section in the image
    words_column1_height = sum(words_font.getsize(word)[1] + grid_padding for word in words[::2])
    words_column2_height = sum(words_font.getsize(word)[1] + grid_padding for word in words[1::2])
    words_section_height = max(words_column1_height, words_column2_height)

    # Calculate the total image height based on the grid size, words section, and padding
    img_height = len(grid) * cell_size + 100 * resolution + words_section_height + grid_padding * 3 + 2 * padding

    # Draw the first image without highlighting words
    img1 = draw_image(title_pos, word_coordinates)

    # Save the image as PNG and/or JPG based on the include_png and include_jpg flags
    if include_png:
        img1.save(png_filename)

    if include_jpg:
        img1.save(jpg_filename)

    # Check if highlighting of words is required
    if highlight:
        # Draw the second image with words in the grid highlighted
        img2 = draw_image(title_pos, words_in_grid=word_coordinates, highlight_words=True, show_word_list=not word_list_highlighted_switch)

        # Save the highlighted image as PNG and/or JPG with filenames starting with "highlighted_"
        if include_png:
            img2.save(f"highlighted_{png_filename}")

        if include_jpg:
            img2.save(f"highlighted_{jpg_filename}")

def grid_to_svg(*kwargs):

    # The variable `formatted_colour` is a formatted string representing a color
    formatted_colour = f"rgb{colour}"
    
    # The function `strip_font_extension` is used to remove file extensions from the `font_choice` variable.
    font = strip_font_extension(font_choice)

    # Responsible for generating the actual SVG based on the grid, words, and various settings.
    def draw_svg(grid, words, filename_suffix, word_coordinates, word_list_visible):

        # Set the column factor
        column_increase_factor = 1.5

        # Calculate the number of rows and columns in the grid.
        rows = len(grid)
        cols = len(grid[0])

        # Calculate the maximum letter width and height based on `letter_size`.
        max_letter_width = letter_size * 0.8
        max_letter_height = letter_size
        
        # Calculate the size of each cell in the SVG grid.
        cell_size = max(max_letter_width, max_letter_height) + 10
        padding = 10

        # Calculate the padding for the header section based on the `title_switch` setting.
        header_padding = 0 if title_switch else padding_title
        
        # Calculate the height of the words section based on the number of words and the `wordlist_size`.
        word_height = wordlist_size
        words_section_height = ((len(words) + 1) // 2) * (word_height + padding)
        
        # Calculate the overall width and height of the SVG based on the grid size, padding, and other settings.
        width = cols * cell_size + 2 * padding
        height = rows * cell_size + 2 * padding + header_padding + words_section_height + padding_word_list

        # Dictionary to determine the number of columns in the words section based on the chosen format.
        format_columns_count = {
            "Comma separated list": 1,
            "Single column": 1,
            "Double column": 2,
            "Triple column": 3,
            "Quad column": 4
        }

        # Create an SVG drawing object using the `svgwrite` library with the calculated width and height
        if word_coordinates:
            # If `word_coordinates` is True, create an SVG with correct filename.
            highlight_file_name = "highlighted_" + svg_filename[:-4] + filename_suffix
            dwg = svgwrite.Drawing(highlight_file_name, (width, height))
            generated_files.append(highlight_file_name)
        else:
            # If `word_coordinates` is False, the SVG filename will not include 'highlighted'.
            dwg = svgwrite.Drawing(svg_filename[:-4] + filename_suffix, (width, height))

        if not title_switch:
            # If `title_switch` is False, create a header with the provided `section` text.
            # Calculate the vertical position (y-coordinate) of the header text based on `header_padding`, `title_pos`, and the header's font size.
            title_y_pos = header_padding / 2 + (title_pos / 2)

            # Create a header text element and add it to the SVG drawing.
            header_text = dwg.text(section, insert=(width / 2, title_y_pos), text_anchor='middle', alignment_baseline='central', font_size=header_size, fill='black', font_family=font)
            
            # Translate the header text element to add padding below it, and then add it to the SVG drawing.
            header_text.translate(0, padding)
            dwg.add(header_text)

        # Create a set to keep track of cells corresponding to words that need to be highlighted.
        highlighted_cells = set()

        # If `word_coordinates` is True and the `highlight_style` is not 'Outline words' or 'Strikethrough words',
        # then iterate through the word_coordinates dictionary to collect the cells that need to be highlighted.
        if word_coordinates and not highlight_style == 'Outline words' or word_coordinates and not highlight_style == 'Strikethrough words':
            for word_cells in word_coordinates.values():
                for cell in word_cells:
                    highlighted_cells.add(cell)

        # Iterate through the rows and columns of the grid to draw the SVG elements for each cell.
        for r in range(rows):
            for c in range(cols):
                char = grid[r][c]
                
                # Convert the character to uppercase or lowercase based on the `make_uppercase` setting.
                if make_uppercase:
                    char = char.upper()
                else:
                    char = char.lower()

                # Calculate the x and y coordinates for placing the cell and its content in the SVG.
                x = c * cell_size + padding
                y = r * cell_size + padding + header_padding
                
                # Create a white-filled rectangle representing the cell and add it to the SVG drawing.
                rect = dwg.rect(insert=(x, y), size=(cell_size, cell_size), fill='white')
                dwg.add(rect)

                # Add the character (letter) to the cell if it is not an empty string.
                if char != "":
                    # Determine the font color for the character based on whether the cell is in the `highlighted_cells` set.
                    font_color = formatted_colour if (r, c) in highlighted_cells else 'black'
                    
                    # If `highlight_style` is 'Strikethrough words' or 'Outline words', set the font color to black to avoid highlighting individual letters.
                    if highlight_style == 'Strikethrough words' or highlight_style == 'Outline words':
                        font_color = 'black'
                    
                    # Create a text element for the character and add it to the SVG drawing at the center of the cell.
                    text = dwg.text(char, insert=(x + cell_size / 2, y + cell_size / 2), text_anchor='middle', alignment_baseline='central', font_size=letter_size, fill=font_color, font_family=font)
                    dwg.add(text)

        def draw_mask_border(dwg, grid, cell_size, padding, header_padding):
            # This function is responsible for drawing borders around the image
            
            # Set the stroke width for the border lines.
            stroke_width = 2
            
            # Get the width and height of the grid from the given `grid` parameter.
            grid_width = len(grid[0])
            grid_height = len(grid)

            # Define a function to check if a cell at position (x, y) in the grid is empty (contains a space).
            def is_empty_cell(x, y):
                return grid[y][x] == " "

            # Define the possible directions (up, down, left, right) to check for neighboring empty cells.
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            # Iterate through each cell in the grid.
            for i in range(grid_height):
                for j in range(grid_width):
                    # If the cell is not empty, check its neighboring cells for potential mask borders.
                    if not is_empty_cell(j, i):
                        for direction in directions:
                            dx, dy = direction
                            x, y = j + dx, i + dy

                            # Check if the neighboring cell is within the grid bounds and empty.
                            if 0 <= x < grid_width and 0 <= y < grid_height and is_empty_cell(x, y):
                                # Based on the direction, determine the start and end points for the mask border line.
                                if dx == 1:
                                    start = (padding + j * cell_size + cell_size, header_padding + i * cell_size)
                                    end = (padding + j * cell_size + cell_size, header_padding + (i + 1) * cell_size)
                                elif dx == -1:
                                    start = (padding + j * cell_size, header_padding + i * cell_size)
                                    end = (padding + j * cell_size, header_padding + (i + 1) * cell_size)
                                elif dy == 1:
                                    start = (padding + j * cell_size, header_padding + i * cell_size + cell_size)
                                    end = (padding + (j + 1) * cell_size, header_padding + i * cell_size + cell_size)
                                elif dy == -1:
                                    start = (padding + j * cell_size, header_padding + i * cell_size)
                                    end = (padding + (j + 1) * cell_size, header_padding + i * cell_size)

                                # Create a line element representing the mask border and add it to the SVG drawing.
                                line = dwg.line(start=start, end=end, stroke='black', stroke_width=stroke_width)
                                dwg.add(line)

        # Check if `mask_border` is True, and if so, call the `draw_mask_border` function to draw the mask borders.
        if mask_border:
            draw_mask_border(dwg, grid, cell_size, padding, header_padding + 10)

        # If `highlight` is True, apply the chosen highlight style to the words in the grid.
        if highlight:
            
            # For 'Outline words' style, outline the rectangular area encompassing each word.
            if highlight_style == 'Outline words':

                for word, word_cells in word_coordinates.items():
                    # Find the minimum and maximum row and column indices for the word's cells.
                    min_row = min(c[0] for c in word_cells)
                    min_col = min(c[1] for c in word_cells)
                    max_row = max(c[0] for c in word_cells)
                    max_col = max(c[1] for c in word_cells)

                    # Check if the word is horizontal, vertical, or diagonal.
                    is_horizontal = min_row == max_row
                    is_vertical = min_col == max_col
                    is_diagonal = not (is_horizontal or is_vertical)

                    if not is_diagonal:
                        # For horizontal or vertical words, draw a rectangle around the word.
                        outline_x = min_col * cell_size + padding
                        outline_y = min_row * cell_size + padding + header_padding
                        outline_width = (max_col - min_col + 1) * cell_size
                        outline_height = (max_row - min_row + 1) * cell_size

                        # Create a rectangle with the outline and add it to the SVG drawing.
                        dwg.add(dwg.rect(insert=(outline_x, outline_y), size=(outline_width, outline_height), fill='none', stroke=formatted_colour, stroke_width=border_width, rx=1, ry=1))
                    else:
                        # For diagonal words, determine the direction (top-left to bottom-right or top-right to bottom-left).

                        # Check if all cells in the word have the same diagonal pattern (top-left to bottom-right).
                        is_top_left_to_bottom_right = all((r - min_row == c - min_col) for r, c in word_cells)

                        # Check if all cells in the word have the same diagonal pattern (top-right to bottom-left).
                        is_top_right_to_bottom_left = all((max_row - r == c - min_col) for r, c in word_cells)

                        # Draw the polyline for the diagonal word in the appropriate direction.
                        offset = cell_size // 2
                        extension_start = 0  # Adjust this value to control the length of the outline at the start
                        extension_end = 0.5   # Adjust this value to control the length of the outline at the end
                        header_padding_offset = header_padding + 35

                        if is_top_left_to_bottom_right:
                            # For top-left to bottom-right diagonal words, draw a polyline around the word.
                            points = [
                                ((min_col - extension_start) * cell_size + padding + offset, (min_row - extension_start) * cell_size + padding + header_padding),
                                ((max_col + extension_end) * cell_size + padding + cell_size - offset, (max_row + extension_end) * cell_size + padding + header_padding),
                                ((max_col + extension_end) * cell_size + padding, (max_row + extension_end + 1) * cell_size + padding - offset + header_padding),
                                ((min_col - extension_start) * cell_size + padding, (min_row - extension_start + 1) * cell_size + padding - offset + header_padding),
                                ((min_col - extension_start) * cell_size + padding + offset, (min_row - extension_start) * cell_size + padding + header_padding)
                            ]
                            # Create a polyline with the outline and add it to the SVG drawing.
                            dwg.add(dwg.polyline(points, stroke=formatted_colour, fill='none', stroke_width=border_width, stroke_linejoin='round'))

                        if is_top_right_to_bottom_left:
                            # For top-right to bottom-left diagonal words, draw a polyline around the word.
                            points = [
                                ((min_col - extension_start) * cell_size + padding + offset, (max_row + extension_start) * cell_size + padding + header_padding_offset),
                                ((max_col + extension_end) * cell_size + padding + cell_size - offset, (min_row - extension_end) * cell_size + padding + header_padding_offset),
                                ((max_col + extension_end) * cell_size + padding, (min_row - extension_end) * cell_size + padding - offset + header_padding_offset),
                                ((min_col - extension_start) * cell_size + padding, (max_row + extension_start) * cell_size + padding - offset + header_padding_offset),
                                ((min_col - extension_start) * cell_size + padding + offset, (max_row + extension_start) * cell_size + padding + header_padding_offset)
                            ]
                            # Create a polyline with the outline and add it to the SVG drawing.
                            dwg.add(dwg.polyline(points, stroke=formatted_colour, fill='none', stroke_width=border_width, stroke_linejoin='round'))

            # For 'Strikethrough words' style, draw lines through the center of the word's cells.
            elif highlight_style == 'Strikethrough words':

                for word, word_cells in word_coordinates.items():
                    # Check if the word follows a snaking path.
                    is_snaking = is_snaking_path(word_cells)

                    if not is_snaking:
                        # If the word is not snaking, find the minimum and maximum row and column indices for the word's cells.
                        min_row = min(c[0] for c in word_cells)
                        min_col = min(c[1] for c in word_cells)
                        max_row = max(c[0] for c in word_cells)
                        max_col = max(c[1] for c in word_cells)

                        # Determine if the word is horizontal, vertical, or diagonal.
                        is_horizontal = min_row == max_row
                        is_vertical = min_col == max_col
                        is_diagonal = not (is_horizontal or is_vertical)

                        if is_horizontal:
                            # For horizontal words, draw a line through the center of the word's cells horizontally.
                            points = [
                                (min_col * cell_size + padding, min_row * cell_size + padding + header_padding + cell_size // 2),
                                ((max_col + 1) * cell_size + padding, min_row * cell_size + padding + header_padding + cell_size // 2)
                            ]
                            # Create the line and add it to the SVG drawing.
                            dwg.add(dwg.line(start=points[0], end=points[1], stroke=formatted_colour, stroke_width=border_width, stroke_linecap='round'))

                        elif is_vertical:
                            # For vertical words, draw a line through the center of the word's cells vertically.
                            points = [
                                (min_col * cell_size + padding + cell_size // 2, min_row * cell_size + padding + header_padding),
                                (max_col * cell_size + padding + cell_size // 2, (max_row + 1) * cell_size + padding + header_padding)
                            ]
                            # Create the line and add it to the SVG drawing.
                            dwg.add(dwg.line(start=points[0], end=points[1], stroke=formatted_colour, stroke_width=border_width, stroke_linecap='round'))

                        elif is_diagonal:
                            # For diagonal words, determine the direction (top-left to bottom-right or top-right to bottom-left).

                            # Check if all cells in the word have the same diagonal pattern (top-left to bottom-right).
                            is_top_left_to_bottom_right = all((r - min_row == c - min_col) for r, c in word_cells)

                            # Check if all cells in the word have the same diagonal pattern (top-right to bottom-left).
                            is_top_right_to_bottom_left = all((max_row - r == c - min_col) for r, c in word_cells)

                            if is_top_left_to_bottom_right:
                                # For top-left to bottom-right diagonal words, draw a line through the center of the word's cells diagonally.
                                points = [
                                    (min_col * cell_size + padding, min_row * cell_size + padding + header_padding),
                                    ((max_col + 1) * cell_size + padding, (max_row + 1) * cell_size + padding + header_padding),
                                ]
                                # Create the line and add it to the SVG drawing.
                                dwg.add(dwg.line(start=points[0], end=points[1], stroke=formatted_colour, stroke_width=border_width, stroke_linecap='round'))

                            if is_top_right_to_bottom_left:
                                # For top-right to bottom-left diagonal words, draw a line through the center of the word's cells diagonally.
                                points = [
                                    (min_col * cell_size + padding, (max_row + 1) * cell_size + padding + header_padding),
                                    ((max_col + 1) * cell_size + padding, min_row * cell_size + padding + header_padding),
                                ]
                                # Create the line and add it to the SVG drawing.
                                dwg.add(dwg.line(start=points[0], end=points[1], stroke=formatted_colour, stroke_width=border_width, stroke_linecap='round'))
                    else:
                        # For snaking path words, draw lines through the center of the word's cells to follow the snaking path.
                        for i, (r, c) in enumerate(word_cells):
                            if i < len(word_cells) - 1:
                                # Draw a line connecting the center of the current cell to the center of the next cell.
                                next_r, next_c = word_cells[i + 1]
                                point1 = (c * cell_size + 0.5 * cell_size + padding, r * cell_size + padding + header_padding + 0.5 * cell_size)
                                point2 = (next_c * cell_size + 0.5 * cell_size + padding, next_r * cell_size + padding + header_padding + 0.5 * cell_size)
                                # Create the line and add it to the SVG drawing.
                                dwg.add(dwg.line(start=point1, end=point2, stroke=formatted_colour, stroke_width=border_width, stroke_linecap='round'))
              
        # Check if gridlines or borders are selected
        if gridlines or border:
            # Draw row lines based on the condition
            for r in range(rows + 1):
                if r == 0 or r == rows or gridlines:
                    # Add line to drawing
                    dwg.add(dwg.line(start=(padding, r * cell_size + padding + header_padding), end=(cols * cell_size + padding, r * cell_size + padding + header_padding), stroke='black'))
            # Draw column lines based on the condition
            for c in range(cols + 1):
                if c == 0 or c == cols or gridlines:
                    # Add line to drawing
                    dwg.add(dwg.line(start=(c * cell_size + padding, header_padding + padding), end=(c * cell_size + padding, rows * cell_size + header_padding + padding), stroke='black'))

        # Define function to add text to columns in the grid
        def add_text_column(dwg, column_x, starting_y, words, font_size, font_family, padding_y):
            column_y = starting_y
            for word in words:
                # Determine the x coordinate for the text
                text_x = column_x + column_width // 2
                # Create text element
                text = dwg.text(word, insert=(text_x, column_y), text_anchor='middle', font_size=font_size, fill='black', font_family=font_family)
                # Add text to drawing
                dwg.add(text)
                # Update the y axis coordinate for next word
                column_y += word_height + padding_y
            return column_y

        # Check before creating a word list
        if not word_list_switch and word_list_visible:
            # Filter words for final_words
            final_words = filter_words(words, rejected_words)
            # Determine the starting y axis for words
            words_start_y = height - words_section_height + padding + word_list_pos

            # Check for required word list format
            if word_list_format in ["Comma separated list", "Single column"]:
                # Set padding_y, and determine column_width
                padding_y = padding
                column_width = (width - padding * 2) * ((word_list_format == "Comma separated list") and column_increase_factor or 1)
                # Wrap words accordingly
                wrapped_words = wrap_text(comma_separated_list(final_words), wordlist_size, column_width) if word_list_format == "Comma separated list" else final_words
                # Change in height after adding text to column
                new_height = height + add_text_column(dwg, (width - column_width) // 2, words_start_y, wrapped_words, wordlist_size, font, padding_y) - height

            # Execute in case of multiple columns
            else:
                # Decide number of columns, words per column, size of column, change in height after addition, and a new start_y
                column_num = format_columns_count[word_list_format]
                words_per_col = len(final_words) // column_num
                padding_y = padding
                column_width = (width - (padding * (column_num + 1))) // column_num
                new_height = height + ((words_per_col * (word_height + padding_y)) + padding_y) - words_section_height
                words_start_y = new_height - (words_per_col * (word_height + padding_y)) - padding

                # Loop over the columns
                for col_num in range(column_num):
                    # Determine the start and end of a column
                    col_start = col_num * words_per_col
                    col_end = (col_num + 1) * words_per_col
                    if col_num == column_num - 1:
                        col_end = None
                    # Calculate x coordinate for column
                    column_x = padding * (col_num + 1) + column_width * col_num
                    # Add text to column
                    add_text_column(dwg, column_x, words_start_y, final_words[col_start:col_end], wordlist_size, font, padding_y)

            # Set the new height of the drawing
            dwg.attribs['height'] = f'{new_height}px'

        # Save the drawing
        dwg.save()

    # Draw the SVG with the given parameters
    draw_svg(grid, words, '.svg', {}, True)

    # If highlighting is required, then draw SVG again with different parameters
    if highlight:
        draw_svg(grid, words, '.svg', word_coordinates, not word_list_highlighted_switch)

# Define a function named parse_sections that takes a file or file path as input.
def parse_sections(file_or_path):
    # Create a defaultdict with a default value of an empty list. This will be used to store the sections and their corresponding content.
    sections = defaultdict(list)
    # Initialize a variable 'current_section' to store the current section name being processed.
    current_section = ""

    # Check if the input is a string representing a file path.
    if isinstance(file_or_path, str):
        # If it is, open the file in read mode and assign it to the variable 'file'.
        file = open(file_or_path, 'r')
    # Check if the input is a file object.
    elif hasattr(file_or_path, 'read'):
        # If it is, simply assign it to the variable 'file'.
        file = file_or_path
    # If the input is neither a file object nor a file path string, raise a ValueError indicating the incorrect input type.
    else:
        raise ValueError("The input must be a file object or a file path string")

    # Use the 'with' statement to open and automatically close the file after processing.
    with file:
        # Iterate over each line in the file using readlines() method.
        for line in file.readlines():
            # Remove leading and trailing whitespace characters from the line and assign it to 'decoded_line'.
            decoded_line = line.strip()

            # Check if the type of 'decoded_line' is bytes (indicating the file is in bytes format).
            if type(decoded_line) == bytes:
                # If it is, decode 'decoded_line' using utf-8 encoding to convert it into a string.
                decoded_line = decoded_line.decode('utf-8')

            # Check if 'decoded_line' starts with the "#" character, indicating the start of a new section.
            if decoded_line.startswith("#"):
                # If it does, set 'current_section' to the content of 'decoded_line' excluding the "#" character.
                current_section = decoded_line[1:]
            # If 'decoded_line' is not empty and does not start with "#", indicating content of a section.
            elif decoded_line:
                # Convert 'decoded_line' to lowercase and append it to the list corresponding to the current section in the 'sections' dict.
                sections[current_section].append(decoded_line.lower())

    # Return the dictionary containing the sections and their content.
    return sections

# Page config.
st.set_page_config(page_title="Word Search Generator", layout="wide")
st.title("Word Search Generator")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Initialize a list for file names.
generated_files = []

# List of words for profanity filter.
bad_words = ['anal', 'anus', 'arse', 'ass', 'assfuck', 'assfucker', 'asshole', 'assshole', 'bastard', 
             'bitch', 'blackcock', 'bloodyhell', 'boong', 'cock', 'cockfucker', 'cocksuck', 'cocksucker', 
             'coon', 'coonnass', 'cunt', 'cyberfuck', 'dick', 'dirty', 'douche', 'dummy', 'erect', 'erection', 'erotic', 
             'escort', 'fag', 'faggot', 'fuck', 'fuckoff', 'fuckyou', 'fuckass', 'fuckhole', 'gook', 'hardcore', 'homoerotic', 
             'lesbian', 'lesbians', 'motherfuck', 'motherfucker', 'negro', 'nigger', 'orgasim', 'orgasm', 'penis', 'penisfucker', 
             'piss', 'pissoff', 'porn', 'porno', 'pornography', 'pussy', 'retard', 'sadist', 'sex', 'sexy', 'shit', 'slut', 'tits', 'viagra', 'whore']

# File type flags.
include_png = False
include_jpg = False
include_svg = False

# create an info box
with st.expander("See info"):

    st.write("### Thanks for visiting Wordy!")

    st.write("""
        This website was made using Python, you can view the source [here](https://github.com/dylnbk/grabby).
             
        Check out my personal website [dylnbk.page](https://dylnbk.page).
        
        You can run this app locally by downloading and opening the Grabby.exe found [here](https://link.storjshare.io/s/jxgizi2qhjoqteuofxywilmnth4a/grabby/Grabby.zip).

        To show support, you can ☕ [buy me a coffee](https://www.buymeacoffee.com/dylnbk).

        **CAUTION** 
        - Leaving the number input at zero will download the entire playlist/profile.
        - HQ will grab the highest available quality, which can take quite a while.
        """)

    st.write("***")

    st.write("""
        ##### YouTube
        - Video (MP4) & Audio (MP3) download.
        - Video (MP4) & Audio (MP3) playlist download.
        - Shorts (MP4) download.
         """)
    
    st.write("***")

    st.write("""
        ##### Instagram
        - Single post & Profile download.
        - Web version doesn't always work.
         """)

    st.write("***")

    st.write("""
        ##### TikTok
        - Single video download.
        - Profile download - up to last 30 videos.
         """)

    st.write("***")

    st.write("""
        ##### Reddit
        - Video (MP4) & Audio (MP3) download - will convert videos to audio.
        - Image (JPG) & Gallery download - will grab all images in a post.
         """)

    st.write("***")

    st.write("""
        ##### Twitter
        - Video (MP4) download.
         """)

    st.write("***")

    st.write("""
        ##### Lucky 🤞
        - You can grab from many different places.
        - Powered by yt dlp - for a full list of supported websites visit [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).
         """)

    st.write("")
    st.write("")


# File upload, .txt .png .jpg .ttf or .zip for a collection.
col1, col2 = st.columns([1,1])
with col1:
    txt_file = st.file_uploader("Upload a word list:", type="txt")
    img_file = st.file_uploader("Upload an image mask:", type=['png', 'jpg'])
with col2:
    font_file = st.file_uploader("Upload a font file:", type="ttf")
    collection_file = st.file_uploader("Upload a collection:", type="zip")
    

# If a txt file is present or collection has been uploaded, generate the form to provide options.
if txt_file is not None or collection_file is not None:

    st.subheader("Text options:")

    # If a font file has been provided.
    if font_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(font_file.read())
            font_choice = tmp_file.name
    # Use a default font type.
    else:
        font_choice = st.selectbox("Font:", ('Arial (SVG)', 'ArchivoBlack-Regular', 'Agency', 'Alger', 'Arimo-Regular', 'Barlow-Regular', 'Calibri (SVG)',
                                                'Comic', 'DMSans-Regular', 'FiraSans-Regular', 'FrederickatheGreat-Regular', 'Georgia (SVG)', 'Heebo-Regular', 
                                                'HindSiliguri-Regular', 'IBMPlexSans-Regular', 'Impact (SVG)', 'Inconsolata-Regular', 'Inter-Regular', 
                                                'JosefinSans-Regular', 'Kanit-Regular', 'Karla-Regular', 'Lato-Regular', 
                                                'LibreBaskerville-Regular', 'LilitaOne-Regular', 'LondrinaShadow-Regular', 'Lora-Regular', 'Lucida',
                                                'Manrope-Regular', 'Merriweather-Regular', 'Monoton-Regular', 'Montserrat-Regular', 
                                                'Mukta-Regular', 'Mulish-Regular', 'NanumGothic-Regular', 'NotoSans-Regular', 'NotoSansJP-Regular',
                                                'NotoSansKR-Regular.otf', 'NotoSansTC-Regular.otf', 'NotoSerif-Regular', 'Nunito-Regular', 'NunitoSans-Regular',
                                                'OpenSans-Regular', 'Oswald-Regular', 'PlayfairDisplay-Regular', 'Poppins-Regular', 'PTSans-Regular',
                                                'PTSerif-Regular', 'Quicksand-Regular', 'Raleway-Regular', 'Roboto', 'RobotoCondensed-Regular',
                                                'RobotoMono-Regular', 'RobotoSlab-Regular', 'Rubik-Regular', 'SecularOne-Regular', 'SourceSansPro-Regular',
                                                'SourceSerifPro-Regular', 'Tahoma (SVG)', 'TimesNewRoman', 'TitilliumWeb-Regular', 'Ubuntu', 'WorkSans-Regular', 'Verdana (SVG)'))
        
        font_choice = "./fonts/" + font_choice

    # Wordsearch options.
    col4, col5, col15 = st.columns([1,1,1])
    with col15:
        language = st.selectbox("Alphabet:", ('English', 'German', 'Greek', 'Italian', 'Spanish', 'Numbers'))
        wordlist_size = st.number_input("Word list size:", value=24, min_value=1, step=1)

    with col4:
        bad_words_toggle = st.selectbox("Profanity filter:", ('On', 'Off'))
        header_size = st.number_input("Header size:", value=24, min_value=1, step=1)
        
    with col5:
        bad_words_sensitivity = st.number_input("Attempts to generate grid:", min_value=1, step=1)
        letter_size = st.number_input("Letter size:", value=24, min_value=1, step=1)
    
    st.subheader("Grid options:")

    col13, col14 = st.columns([1,1])
    with col13:
        rows = st.number_input("Grid rows:", value=15, min_value=1, step=1)
        threshold = st.number_input("Threshold:", value=100, min_value=1, max_value=250, step=1)
    with col14:    
        cols = st.number_input("Grid columns:", value=15, min_value=1, step=1)
        resolution = st.number_input("Resolution:", min_value=1, step=1)

    col6, col7, col8 = st.columns([1,1,1])
    
    with col6:
        highlight_style = st.selectbox("Generate solution:", ('None', 'Highlight letters', 'Outline words', 'Strikethrough words'))
        border_width = st.number_input("Outline thickness:", value=2, min_value=1)
        make_uppercase = st.checkbox("Capitalize letters")
        word_list_switch = st.checkbox("Remove word list")
        border = st.checkbox("Show border")
        
    with col7:
        shape = st.selectbox("Grid shape:", ('Square', 'Circle', 'Triangle', 'Pentagon', 'Hexagon', 'Star', 'Heart'))
        word_list_pos = st.number_input("Word list position:", value=1, min_value=1, step=1)
        word_list_highlighted_switch  = st.checkbox("Remove solved word list")
        mask_border = st.checkbox("Image mask border")
        show_image = st.checkbox("Show image")
        
    with col8:
        word_list_format = st.selectbox("Word list style:", ("Comma separated list", "Single column", "Double column", "Triple column", "Quad column"))
        title_pos = st.number_input("Title position:", value=1, min_value=1, step=1)
        title_switch = st.checkbox("Remove title")
        gridlines = st.checkbox("Show gridlines")
        grayscale_toggle = st.checkbox("Grayscale image")

    if highlight_style == 'None':
        highlight = False
    else:
        highlight = True

    image_opacity = st.slider("Image opacity:", min_value=1, max_value=255, step=1)
    mask_border_opacity = st.slider("Border opacity:", min_value=1, max_value=255, step=1)
    mask_border_opacity = 255 - mask_border_opacity
    hex_code = st.color_picker('Pick a highlight color:', '#565656')
    image_formats = st.multiselect(
        'Image formats:',
        ['.png', '.jpg', '.svg'],
        ['.png'])
        
    colour = hex_to_rgb(hex_code)
    padding_title = 60
    padding_word_list = 50

    
    # Word direction choices --------------------------------------
    st.write('---')
    st.write("Word directions:")
    col9, col10, col11, col12 = st.columns([0.7,1,1,1])
    st.write('---')

    with col9:
        all_directions = st.checkbox("All directions")
        st.write("")

    if all_directions:

        with col10:
            ru = st.checkbox("↖", value=True)
            rh = st.checkbox("←	", value=True)
            rur = st.checkbox("↙", value=True)
        
        with col11:
            rv = st.checkbox("↑", value=True)
            snake = st.checkbox("↝")
            v = st.checkbox("↓", value=True)

        with col12:
            ud = st.checkbox("↗", value=True)
            h = st.checkbox("→", value=True)
            d = st.checkbox("↘", value=True)

    else:
        
        with col10:
            ru = st.checkbox("↖")
            rh = st.checkbox("←	")
            rur = st.checkbox("↙")
        
        with col11:
            rv = st.checkbox("↑")
            snake = st.checkbox("↝")
            v = st.checkbox("↓")

        with col12:
            ud = st.checkbox("↗")
            h = st.checkbox("→")
            d = st.checkbox("↘")

    word_directions = sort_word_directions(rur, rh, ru, rv, v, ud, h, d, snake)
    # Word direction choices end --------------------------------------

    # Invert the threshold selection.
    threshold = 250 - threshold

    # When user presses 'Generate word searches'
    if st.button("Generate Word Searches"):

        try:
            
            # Update file type flags.
            if '.png' in image_formats:
                include_png = True
            
            if '.jpg' in image_formats:
                include_jpg = True

            if '.svg' in image_formats:
                include_svg = True

            # Check if the user has uploaded a collection file.
            if collection_file is not None:
                
                # Read the contents of the uploaded file and create a BytesIO object for further processing.
                collection_contents = collection_file.read()
                collection_obj = BytesIO(collection_contents)
                
                # Unzip the collection file and get a list of filenames inside it.
                collection_file_names = unzip_file(collection_obj)
                
                # Initialize empty lists to store image filenames and text file filenames separately.
                collection_images = []
                collection_words = []
                
                # Iterate through each filename obtained from the unzipped collection.
                for filename in collection_file_names:
                    # Get the file extension of the current filename.
                    extension = filename.split('.')[-1]
                    
                    # Check if the file is an image by comparing its extension to the image file extensions.
                    if extension.lower() in ['jpg', 'jpeg', 'png']:
                        collection_images.append(filename)
                    
                    # Check if the file is a text file by comparing its extension to the text file extension.
                    elif extension.lower() in ['txt']:
                        collection_words.append(filename)

                # Sort the lists of image filenames and text file filenames separately based on trailing numbers.
                collection_images.sort()
                collection_words.sort()
                
                # Display an empty line in the output.
                st.write("")

                # Create a progress bar to show the progress of generating word search grids.
                progress_bar = st.progress(0, text=None)
                
                # Iterate through each text file in the collection_words list.
                for index, value in enumerate(collection_words):
                    # Parse the sections and content from the current text file and store them in a dictionary.
                    sections = parse_sections(collection_words[index])
                    
                    # Get the file path of the corresponding image file for this text file.
                    img_file_path = f"./{collection_images[index]}"

                    # Iterate through each section and its corresponding words in the text file.
                    for idx, (section, words) in enumerate(sections.items(), start=1):
                        # Generate the word search grid, rejected words, and word coordinates based on the given parameters.
                        grid, rejected_words, word_coordinates = generate_word_search(words, rows, cols, img_file_path, threshold, word_directions, language, bad_words, bad_words_sensitivity, bad_words_toggle)

                        # If a valid grid is generated, proceed with creating word search images in different formats.
                        if grid:
                            # Set filenames for PNG and JPG formats based on the index and section number.
                            png_filename = f"word_search_{index}_{idx}.png"
                            jpg_filename = f"word_search_{index}_{idx}.jpg"
                            
                            # Convert the word search grid to PNG and JPG images, and optionally, generate highlighted versions.
                            grid_to_image(grid, words, rejected_words, png_filename, jpg_filename, section,
                                            font_choice, header_size, wordlist_size, letter_size, highlight,
                                            resolution, make_uppercase, gridlines, border, title_switch,
                                            word_list_switch, word_list_highlighted_switch, padding_title,
                                            padding_word_list, colour, word_list_format, mask_border, show_image,
                                            img_file_path, image_opacity, word_coordinates, highlight_style,
                                            border_width, title_pos, word_list_pos, include_png, include_jpg,
                                            grayscale_toggle)

                            # Add generated filenames to the list 'generated_files' based on the user's preferences.
                            if include_png:
                                generated_files.append(png_filename)

                                if highlight:
                                    generated_files.append(f"highlighted_{png_filename}")

                            if include_jpg:
                                generated_files.append(jpg_filename)

                                if highlight:
                                    generated_files.append(f"highlighted_{jpg_filename}")

                            # If the user chooses to include SVG format, generate the word search grid in SVG format and, if requested, create a highlighted version.
                            if include_svg:

                                # Generate name.
                                svg_filename = f"word_search_{index}_{idx}.svg"

                                # Create an SVG grid image
                                grid_to_svg(grid, words, rejected_words, svg_filename, section, font_choice, header_size, wordlist_size,
                                    letter_size, highlight, make_uppercase, gridlines, border, title_switch, word_list_switch,
                                    word_list_highlighted_switch, padding_title, padding_word_list, colour, word_list_format,
                                    mask_border, highlight_style, border_width, title_pos, word_list_pos, word_coordinates)
                                
                                # Save the filename
                                generated_files.append(svg_filename)
                    
                    # Update progress bar to indicate the completion percentage of word search grid generation.
                    progress_bar.progress((index + 1) / len(collection_words))

                # After generating all word search grids, create a ZipFile named "output.zip" and add the generated files to it.
                with ZipFile(f"output.zip", 'w') as zipObj:
                    for file in generated_files:
                        zipObj.write(file)

                # Create a download button for the user to download the generated ZipFile.
                with open(f"output.zip", "rb") as file:
                    st.download_button("Download", data=file, file_name=f"output.zip", mime="zip")

                # Display the generated PNG images to the user.
                for file in generated_files:
                    if ".png" in file:
                        st.image(file)

                # Delete all the generated files to clean up after the process.
                for file in generated_files:
                    delete_files(file)

                # Also delete the ZipFile "output.zip".
                delete_files(f"output.zip")

                # Clean up the image and text files used for generating word search grids.
                for image_file in collection_images:
                    delete_files(image_file)

                for word_file in collection_words:
                    delete_files(word_file)

            # A single text file has been used
            else:
                # Parse the text file into sections
                sections = parse_sections(txt_file)

                # If an image file has been provided, get its path
                if img_file is not None:
                    img_file_path = image_path_find(img_file)
                else:
                    # Set the image file path based on the given shape
                    if shape == 'Square':
                        img_file_path = "./default_shapes/square.jpg"
                    elif shape == 'Circle':
                        img_file_path = "./default_shapes/circle.jpg"
                    elif shape == 'Triangle':
                        img_file_path = "./default_shapes/triangle.jpg"
                    elif shape == 'Pentagon':
                        img_file_path = "./default_shapes/pentagon.jpg"
                    elif shape == 'Hexagon':
                        img_file_path = "./default_shapes/hexagon.jpg"
                    elif shape == 'Star':
                        img_file_path = "./default_shapes/star.jpg"
                    elif shape == 'Heart':
                        img_file_path = "./default_shapes/heart.jpg"

                st.write("")
                # Initialize progress bar
                progress_bar = st.progress(0, text=None)

                # Loop through each section in order to generate word search grid
                for idx, (section, words) in enumerate(sections.items(), start=1):
                    grid, rejected_words, word_coordinates = generate_word_search(words, rows, cols, img_file_path, threshold, word_directions, 
                                                                                language, bad_words, bad_words_sensitivity, 
                                                                                bad_words_toggle)
                    # If grid is successfully created
                    if grid:
                        # Generate name for png and jpg files
                        png_filename = f"word_search_{idx}.png"
                        jpg_filename = f"word_search_{idx}.jpg"

                        # Convert the grid into an image
                        grid_to_image(grid, words, rejected_words, png_filename, jpg_filename, section,
                                    font_choice, header_size, wordlist_size, letter_size, highlight,
                                    resolution, make_uppercase, gridlines, border, title_switch,
                                    word_list_switch, word_list_highlighted_switch, padding_title,
                                    padding_word_list, colour, word_list_format, mask_border, show_image,
                                    img_file_path, image_opacity, word_coordinates, highlight_style,
                                    border_width, title_pos, word_list_pos, include_png, include_jpg,
                                    grayscale_toggle)
                        
                        # Save the files if required
                        if include_png:
                            generated_files.append(png_filename)

                            if highlight:
                                generated_files.append(f"highlighted_{png_filename}")

                        if include_jpg:
                            generated_files.append(jpg_filename)

                            if highlight:
                                generated_files.append(f"highlighted_{jpg_filename}")

                        # Also generate svg file if required
                        if include_svg:
                            svg_filename = f"word_search_{idx}.svg"
                            grid_to_svg(grid, words, rejected_words, svg_filename, section, font_choice, header_size, wordlist_size,
                                        letter_size, highlight, make_uppercase, gridlines, border, title_switch, word_list_switch,
                                        word_list_highlighted_switch, padding_title, padding_word_list, colour, word_list_format,
                                        mask_border, highlight_style, border_width, title_pos, word_list_pos, word_coordinates)
                            
                            generated_files.append(svg_filename)

                    # Update progress bar
                    progress_bar.progress(idx / len(sections))

                # Create a zip file with all the generated files
                with ZipFile(f"output.zip", 'w') as zipObj:

                    for file in generated_files:

                        # Add file to zip
                        zipObj.write(file)

                # Create a download button for the user to download the zip file
                with open(f"output.zip", "rb") as file:
                    st.download_button("Download", data=file, file_name=f"output.zip", mime="zip")

                # Display all generated png files
                for file in generated_files:
                    if ".png" in file:
                        st.image(file)

                # Delete all generated files
                for file in generated_files:

                    delete_files(file)

                # Delete the zip file
                delete_files(f"output.zip")

        except Exception as e:

            st.error(f"Error: {e}")
