
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io  # Add this import
# Define your 2D points as a list of (x, y) tuples
points_list = [(1, 2), (2, 3), (3, 4), (4, 5)]  # Replace with your points

# Create a blank figure
fig, ax = plt.subplots()

# Calculate the min and max values for the x and y axes to set appropriate limits
min_x = min(point[0] for point in points_list)
max_x = max(point[0] for point in points_list)
min_y = min(point[1] for point in points_list)
max_y = max(point[1] for point in points_list)

# Create an animation
output_file = 'animation.gif'
with imageio.get_writer(output_file, mode='I', duration=0.2) as writer:
    for i in range(len(points_list)):
        ax.cla()  # Clear the previous plot
        x, y = zip(*points_list[:i + 1])  # Extract points up to the current frame
        ax.plot(x, y, marker='o', linestyle='-')
        ax.set_xlim(min_x, max_x)  # Set appropriate x-axis limits
        ax.set_ylim(min_y, max_y)  # Set appropriate y-axis limits

        # Save the current frame to the GIF
        # Save the current frame to the GIF
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

plt.close()  # Close the plot to release resources
