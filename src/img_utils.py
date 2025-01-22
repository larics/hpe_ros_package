
#!/opt/conda/bin/python3
import numpy 
import rospy
import cv2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2

def convert_pil_to_ros_img(img):
        img = img.convert('RGB')
        msg = Image()
        stamp = rospy.Time.now()
        msg.height = img.height
        msg.width = img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * img.width
        msg.data = numpy.array(img).tobytes()
        return msg

def convert_pil_to_ros_compressed(img, color_conversion = False, compression_type="jpeg"):

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()       
        msg.format = "{}".format(compression_type)
        np_img = numpy.array(img); #bgr 

        if color_conversion: 
            np_img = bgr2rgb(np_img)            
        
        compressed_img = cv2.imencode(".{}".format(compression_type), np_img)[1]
        msg.data = compressed_img.tobytes()

        return msg

def convert_ros_to_pil_img(msg):
    try:
        rospy.loginfo_once("Converting ROS image to PIL image, encoding is {}")
        width = msg.width
        height = msg.height
        channels = 3 if (msg.encoding == "rgb8" or msg.encoding == "bgr8") else 1  # Assuming RGB or grayscale
        
        # Convert the ROS image data (byte array) to a NumPy array
        image_data = numpy.frombuffer(msg.data, dtype=numpy.uint8)

        if channels == 3: 
            np_img = image_data.reshape((height, width, channels))
            np_img = bgr2rgb(np_img)
        else:
            np_img = image_data.reshape((height, width))
    
        pil_img = PILImage.fromarray(np_img)

        return pil_img

    except Exception as e:
        rospy.logerr("Error converting ROS image to PIL image: {}".format(e))
        return None

def get_text_dimensions(text_string, font):

    ascent, descent = font.getmetrics()
    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return (text_width, text_height)
    
def bgr2rgb(img):
        
    rgb_img = numpy.zeros_like(img)
    rgb_img[:, :, 0] = img[:, :, 2]
    rgb_img[:, :, 1] = img[:, :, 1]
    rgb_img[:, :, 2] = img[:, :, 0]
        
    return rgb_img

def plot_hand_keypoints(image=None, keypoints=None, save_path=None):
    """
    Plot hand keypoints on a given Pillow Image and label them with their index.

    Args:
        image (PIL.Image.Image): Input image. If None, a blank white canvas is created.
        keypoints (list of tuples): List of (x, y) keypoints to plot.
        save_path (str): Path to save the resulting image. If None, the image is not saved.

    Returns:
        PIL.Image.Image: The image with keypoints plotted.
    """
    # Validate inputs
    if image is None:
        # Create a blank white canvas if no image is provided
        width, height = 500, 500
        image = Image.new("RGB", (width, height), "white")
    if keypoints is None:
        raise ValueError("Keypoints list cannot be None.")

    draw = ImageDraw.Draw(image)

    # Load a font (optional, can use default)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)  # Use a custom font if available
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    # Loop through the keypoints
    for idx, (x, y) in enumerate(keypoints, start=1):
        # Draw a circle at each keypoint
        r = 6
        draw.ellipse((x - r/2, y - r/2, x + r/2, y + r/2), fill="red")
        # Draw the index number above the dot
        draw.text((x, y - 10), str(idx), fill="green", font=font)

    # Save the image if a save path is provided
    if save_path:
        image.save(save_path)

    return image
