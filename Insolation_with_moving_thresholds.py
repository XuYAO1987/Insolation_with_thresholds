import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from PIL import Image
import imageio
import os

### STEP 1 ### Creat insolation series with moving thresholds to generate rank and couplet time series
# Read 4 CSV files of insolation, threshold, rank and couplet
file_path1 = 'Insolation_series.csv'  
file_path2 = 'Insolation_threshold_dataset.csv'  
file_path3 = 'Insolation_rank_dataset.csv'  
file_path4 = 'Insolation_couplet_dataset.csv'  
df1 = pd.read_csv(file_path1, header=0)  
df2 = pd.read_csv(file_path2, header=0)  
df3 = pd.read_csv(file_path3, header=0)  
df4 = pd.read_csv(file_path4, header=0)  

# Get number of time series groups
num_series = df2.shape[1] // 2  # Every two columns represent one time series

# Create figure and axes
# Canvas parameters can be modified as needed
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.95, 7))  
ax1.set_ylabel('W/m$^2$', fontsize=12)
ax1.set_title('Insolation with threshold')
ax2.set_ylabel('Rank', fontsize=12)
ax3.set_xlabel('Age (Ka)', fontsize=12)
ax3.set_ylabel('Duration (kyr)', fontsize=12)

# Define color gradient
colors = plt.cm.winter(np.linspace(0, 1, num_series))  # Gradient from blue to green

# Plot time series from File 1 (baseline curve)
ax1.plot(df1.iloc[:, 0], df1.iloc[:, 1], color='gray', lw=0.5)

# Initialize empty lines for time series from Files 2, 3 and 4
line1, = ax1.plot([], [], lw=2, linestyle='--', label='Moving threshold')
line2, = ax2.plot([], [], lw=0.5, label='Rank time series')
line3, = ax3.plot([], [], lw=1, label='Couplet time series')

# Set axis ranges
ax1.set_xlim(-10000, 0)   # X-axis range
ax1.set_ylim(410, 530)    # Y-axis range
ax2.set_xlim(-10000, 0)   # X-axis range
ax2.set_ylim(-2, 2)       # Y-axis range
ax3.set_xlim(-10000, 0)   # X-axis range
ax3.set_ylim(10, 2000)    # Y-axis range
ax3.set_yscale('log')

# Add legends
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')

# Define update function for animation
def update(frame):
    # Get current series group
    current_series = frame % num_series
    
    # Get data for current series group
    x2 = df2.iloc[:, 2 * current_series]  # Time values (every 2i-th column)
    y2 = df2.iloc[:, 2 * current_series + 1]  # Values (every 2i+1-th column)
    x3 = df3.iloc[:, 2 * current_series]  # Time values
    y3 = df3.iloc[:, 2 * current_series + 1]  # Values
    x4 = df4.iloc[:, 2 * current_series]  # Time values
    y4 = df4.iloc[:, 2 * current_series + 1]  # Values
    
    # Update line data and colors
    line1.set_data(x2, y2)
    line1.set_color(colors[current_series])
    line2.set_data(x3, y3)
    line2.set_color(colors[current_series])
    line3.set_data(x4, y4)
    line3.set_color(colors[current_series])
    
    # Update titles
    ax1.set_title(f'Insolation with threshold (W/m$^2$): {current_series + 428}', fontsize=15)
    return line1, line2, line3

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_series, interval=200, blit=True)
# Save as GIF
ani.save('1-Insolation_rank_couplet.gif', writer='pillow')



### STEP 2 ### Creat wavelet coherence analyses of rank and couplet with moving thresholds
def generate_wavelet_coherence(image_folder, output_gif_path):
    # Get sorted list of image files
    image_files = sorted([os.path.join(image_folder, f) 
                         for f in os.listdir(image_folder) 
                         if f.endswith('.png') or f.endswith('.jpg')])
    images = [Image.open(image) for image in image_files]   # Open all images
    # Save as GIF (200ms delay between frames, infinite loop)
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=200, loop=0) 
    return images

#image_folder is current folder which contains wavelet coherence images, and this 
image_folder = Path(__file__).parent if "__file__" in locals() else os.getcwd() 
output_path = '2-Rank_couplet_wavelet_coherence.gif'

# Generate the GIF and store the images in images_new
images_new = generate_wavelet_coherence(image_folder, output_path) 



### STEP 3 ### Combine the insolation, rank, couplet time series with the wavelet coherence results

def combine_gifs_centered_loop(gif1_path, gif2_path, output_path):
    # Read two above GIF files
    gif1 = imageio.get_reader('1-Insolation_rank_couplet.gif')
    gif2 = imageio.get_reader('2-Rank_couplet_wavelet_coherence.gif')
    
    # Get number of frames and dimensions for each GIF
    frames1 = [frame for frame in gif1]
    frames2 = [frame for frame in gif2] 
   
    # Ensure both GIFs have same number of frames
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]
    
    # Get dimensions of each GIF
    width1, height1 = frames1[0].shape[1], frames1[0].shape[0]
    width2, height2 = frames2[0].shape[1], frames2[0].shape[0]
    
    # Calculate combined dimensions
    combined_width = max(width1, width2)
    combined_height = height1 + height2
    
    # Create combined GIF
    with imageio.get_writer(output_path, mode='I', duration=gif1.get_meta_data()['duration'], loop=0) as writer:
        for frame1, frame2 in zip(frames1, frames2):
            # Convert frames to PIL images
            img1 = Image.fromarray(frame1)
            img2 = Image.fromarray(frame2)
            
            # Create new blank image
            combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
            
            # Calculate centering offsets
            offset_x1 = (combined_width - width1) // 2   
            offset_x2 = (combined_width - width2) // 2   
            
            # Paste frames onto new image with center alignment
            combined_image.paste(img1, (offset_x1, 0))
            combined_image.paste(img2, (offset_x2 - 10, height1))  # Adjust parameter to ensure perfect alignment
            
            # Add combined frame to GIF
            writer.append_data(np.array(combined_image))

# Call function to combine GIFs
combine_gifs_centered_loop('gif1.gif', 'gif2.gif', '3-Insolation_imposing_moving_threshold.gif')

# Show GIF
img = Image.open('3-Insolation_imposing_moving_threshold.gif')
img.show() 